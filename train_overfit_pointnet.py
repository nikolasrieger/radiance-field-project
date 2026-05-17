import torch
import os
import torch.nn.functional as F
from torch.optim import AdamW
from torch import nn
from gsplat import rasterization
from uco3d.dataset_utils.utils import opencv_cameras_projection_from_uco3d
from uco3d.dataset_utils.data_types import GaussianSplats
from src.data.datamodule import UCO3DDataModule
from src.utils.splat_visualizer import SplatRenderVisualizer
from src.models.pointnet2_utils import farthest_point_sample, index_points
from models.pointnet2_cls_ssg import get_model as get_pointnet2

os.environ["UCO3D_DATASET_ROOT"] = "/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw"

EPOCHS         = 10000
LR             = 1e-4
NUM_VIEWS      = 10
POINTNET_CKPT  = "/mnt/hdd/lkuech/repos/radiance-field-project/checkpoints/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
NORMAL_CHANNEL = False
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── data ──────────────────────────────────────────────────────────────────────
dm = UCO3DDataModule(batch_size=1, num_workers=0, image_height=512, image_width=512, load_masks=True)
torch.manual_seed(1)
train_loader = dm.train_loader()
print(f"Dataset size: {len(train_loader.dataset)}")

train_iter      = iter(train_loader)
training_frames = [next(train_iter) for _ in range(NUM_VIEWS)]

raw_splats = training_frames[0]["sequence_gaussian_splats"][0]

input_splats = GaussianSplats(
    means     = raw_splats.means.to(DEVICE).detach().float(),
    sh0       = raw_splats.sh0.to(DEVICE).detach().float(),
    scales    = raw_splats.scales.to(DEVICE).detach().float(),
    quats     = raw_splats.quats.to(DEVICE).detach().float(),
    opacities = raw_splats.opacities.to(DEVICE).detach().float(),
    shN       = None,
    fg_mask   = None,
)

# ── camera setup ──────────────────────────────────────────────────────────────
def get_camera_params(sample, device):
    gt_image = sample["image_rgb"].to(device)
    fg_mask  = sample["fg_probability"].to(device)
    h, w     = gt_image.shape[-2:]
    image_size_tensor = torch.tensor([[h, w]], dtype=torch.float32)
    Rcv, tvec, K = opencv_cameras_projection_from_uco3d(
        sample["camera"], image_size=image_size_tensor,
    )
    viewmats = torch.eye(4)[None].to(device)
    viewmats[:, :3, :4] = torch.cat([Rcv, tvec[..., None]], dim=-1)
    K = K.to(device)
    return viewmats, K, gt_image, fg_mask, h, w

view_data = [get_camera_params(f, DEVICE) for f in training_frames]

# ── render ────────────────────────────────────────────────────────────────────
def render(means, quats, scales, opacities, sh0, viewmats, K, h, w):
    imgs, _, _ = rasterization(
        means          = means,
        quats          = F.normalize(quats, p=2, dim=-1),
        scales         = torch.exp(scales),
        opacities      = torch.sigmoid(opacities.flatten()),
        colors         = sh0.unsqueeze(1),
        viewmats       = viewmats,
        Ks             = K,
        width          = w,
        height         = h,
        sh_degree      = 0,
        near_plane     = 0.01,
        backgrounds    = torch.zeros(1, 3, device=DEVICE),
        packed         = False,
        absgrad        = False,
        sparse_grad    = False,
        rasterize_mode = "classic",
        distributed    = False,
    )
    return imgs.permute(0, 3, 1, 2)

def l2_loss(pred, gt, mask=None):
    squared_diff = (pred - gt) ** 2
    if mask is None:
        return squared_diff.mean()
    binary_mask = (mask > 0.5).float()
    return (squared_diff * binary_mask).sum() / binary_mask.sum().clamp(min=1.0)

# ── Multi-scale PointNet++ model ──────────────────────────────────────────────
class MultiScalePointNet2Network(nn.Module):
    """
    Encoder: hierarchical FPS cascade reduces 50k → 4096 → 1024
             before entering the pretrained PointNet++ SA layers.
             This preserves spatial structure at every scale.

    Decoder: broadcast global feat + per-point raw features → residual deltas
             applied to ALL 50k original Gaussians.
    """
    def __init__(self, pointnet_ckpt: str, normal_channel: bool = False):
        super().__init__()

        # Load pretrained PointNet++ and keep only SA layers
        full_model = get_pointnet2(num_class=40, normal_channel=normal_channel)
        ckpt = torch.load(pointnet_ckpt, map_location='cpu', weights_only=False)
        full_model.load_state_dict(ckpt['model_state_dict'])

        self.sa1 = full_model.sa1   # expects ~1024 pts
        self.sa2 = full_model.sa2
        self.sa3 = full_model.sa3
        # → produces (B, 1024) global feature

        # Decoder: global_feat (1024) + per-point splat params (14) → delta (14)
        self.decoder = nn.Sequential(
            nn.Linear(1024 + 14, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 14),
        )

    def encode(self, xyz, normals=None):
        """
        xyz:     (B, N, 3)  — expects N ~= 1024 at this stage
        Returns: (B, 1024)
        """
        xyz_t  = xyz.permute(0, 2, 1)                      # (B, 3, N)
        norm_t = normals.permute(0, 2, 1) if normals is not None else None

        l1_xyz, l1_pts = self.sa1(xyz_t, norm_t)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)

        return l3_pts.view(xyz.shape[0], -1)               # (B, 1024)

    def encode_multiscale(self, means_b, normals=None):
        """
        means_b: (1, N, 3)
        """
        from src.models.pointnet2_utils import farthest_point_sample, index_points

        # ── Normalize to unit sphere (ModelNet40 training scale) ──
        centroid = means_b.mean(dim=1, keepdim=True)          # (1, 1, 3)
        means_centered = means_b - centroid                   # (1, N, 3)
        scale = means_centered.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True).values
        means_normed = means_centered / (scale + 1e-8)        # (1, N, 3) in [-1, 1]

        # ── FPS: 50k → 1024 ──
        idx  = farthest_point_sample(means_normed, 1024)
        pts  = index_points(means_normed, idx)                # (1, 1024, 3)

        # ── SA layers expect (B, 3, N) ──
        xyz_t = pts.permute(0, 2, 1)                          # (1, 3, 1024)

        l1_xyz, l1_pts = self.sa1(xyz_t, None)                # → (1, 3, 512), (1, 128, 512)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)            # → (1, 3, 128), (1, 256, 128)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)            # → (1, 3, 1),   (1, 1024, 1)

        return l3_pts.view(means_b.shape[0], -1)              # (1, 1024)


    def reconstruct_splats(self, splats, normals=None):
        means     = splats.means        # (N, 3)
        sh0       = splats.sh0
        scales    = splats.scales
        quats     = splats.quats
        opacities = splats.opacities    # (N,)

        means_b = means.unsqueeze(0)    # (1, N, 3)

        # Encode with hierarchical FPS — sees all 50k spatially
        global_feat = self.encode_multiscale(means_b, normals)  # (1, 1024)

        N = means.shape[0]
        global_feat_expanded = global_feat.expand(N, -1)        # (N, 1024)

        flat = torch.cat([
            means, sh0, scales, quats, opacities.unsqueeze(-1)
        ], dim=-1)                                               # (N, 14)

        decoder_input = torch.cat([global_feat_expanded, flat], dim=-1)  # (N, 1038)
        delta = self.decoder(decoder_input)                               # (N, 14)

        return GaussianSplats(
            means     = means     + delta[..., 0:3],
            sh0       = sh0       + delta[..., 3:6],
            scales    = scales    + 0.01 * delta[..., 6:9].clamp(-1, 1),
            quats     = quats     + delta[..., 9:13],
            opacities = opacities + delta[..., 13],
            shN       = splats.shN,
            fg_mask   = splats.fg_mask,
        )

# ── model + optimiser ─────────────────────────────────────────────────────────
network   = MultiScalePointNet2Network(
    pointnet_ckpt  = POINTNET_CKPT,
    normal_channel = NORMAL_CHANNEL,
).to(DEVICE)
optimizer = AdamW(network.parameters(), lr=LR)

# ── reference render (no grad) ────────────────────────────────────────────────
v0_mats, v0_K, v0_gt_image, v0_fg_mask, v0_h, v0_w = view_data[0]

with torch.no_grad():
    ref_render = render(
        input_splats.means, input_splats.quats, input_splats.scales,
        input_splats.opacities, input_splats.sh0,
        v0_mats, v0_K, v0_h, v0_w,
    )

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    network.train()
    total_epoch_loss = 0.0

    for v_mats, v_K, v_gt_image, v_fg_mask, v_h, v_w in view_data:
        optimizer.zero_grad()

        recon = network.reconstruct_splats(input_splats)

        rendered = render(
            recon.means, recon.quats, recon.scales,
            recon.opacities, recon.sh0,
            v_mats, v_K, v_h, v_w,
        )

        loss = l2_loss(rendered, v_gt_image, mask=v_fg_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        total_epoch_loss += loss.item()

    if epoch % 20 == 0:
        avg_loss = total_epoch_loss / NUM_VIEWS
        network.eval()
        with torch.no_grad():
            recon_eval = network.reconstruct_splats(input_splats)
            eval_rendered = render(
                recon_eval.means, recon_eval.quats, recon_eval.scales,
                recon_eval.opacities, recon_eval.sh0,
                v0_mats, v0_K, v0_h, v0_w,
            )
            SplatRenderVisualizer.save(
                "gt_vs_recon_pointnet.png",
                render_colors = torch.cat([
                    v0_gt_image.permute(0, 2, 3, 1),
                    ref_render.permute(0, 2, 3, 1),
                    eval_rendered.permute(0, 2, 3, 1),
                ], dim=2),
                render_alphas = torch.ones(1, v0_h, v0_w * 3, 1, device=DEVICE),
            )
        print(f"Epoch {epoch:05d}/{EPOCHS} | Avg Loss = {avg_loss:.6f}")

# ── final visualisation ───────────────────────────────────────────────────────
network.eval()
with torch.no_grad():
    recon_final = network.reconstruct_splats(input_splats)
    final_render = render(
        recon_final.means, recon_final.quats, recon_final.scales,
        recon_final.opacities, recon_final.sh0,
        v0_mats, v0_K, v0_h, v0_w,
    )

SplatRenderVisualizer.save(
    "gt_vs_recon_multiscale.png",
    render_colors = torch.cat([
        v0_gt_image.permute(0, 2, 3, 1),
        ref_render.permute(0, 2, 3, 1),
        final_render.permute(0, 2, 3, 1),
    ], dim=2),
    render_alphas = torch.ones(1, v0_h, v0_w * 3, 1, device=DEVICE),
)
print("Saved gt_vs_recon_multiscale.png")
