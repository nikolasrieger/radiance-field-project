import torch
import os
import torch.nn.functional as F
from torch.optim import AdamW
from torch import nn
from gsplat import rasterization_2dgs
from uco3d.dataset_utils.utils import opencv_cameras_projection_from_uco3d
from uco3d.dataset_utils.data_types import GaussianSplats
from src.data.datamodule import UCO3DDataModule
from src.utils.splat_visualizer import SplatRenderVisualizer

os.environ["UCO3D_DATASET_ROOT"] = "/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw"

EPOCHS    = 10000
LR        = 1e-4
NUM_VIEWS = 10
LOG_EVERY = 50
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── data ──────────────────────────────────────────────────────────────────────
dm = UCO3DDataModule(
    batch_size=1, num_workers=0,
    image_height=512, image_width=512,
    load_masks=True,
)
torch.manual_seed(1)
train_loader    = dm.train_loader()
print(f"Dataset size: {len(train_loader.dataset)}")

train_iter      = iter(train_loader)
training_frames = [next(train_iter) for _ in range(NUM_VIEWS)]

raw_splats = training_frames[0]["sequence_gaussian_splats"][0]

# ── input splats (3DGS from uCO3D) — we initialise 2DGS from these ───────────
# We keep means, sh0, quats, opacities as-is.
# scales: 3DGS has (N,3); we drop the 3rd dim to get a (N,2) 2DGS disk scale.
means_init     = raw_splats.means.to(DEVICE).detach().float()       # (N, 3)
sh0_init       = raw_splats.sh0.to(DEVICE).detach().float()         # (N, 3)
scales_init    = raw_splats.scales.to(DEVICE).detach().float()      # (N, 3)
scales2d_init  = scales_init[:, :2]                                 # (N, 2)  drop depth
quats_init     = raw_splats.quats.to(DEVICE).detach().float()       # (N, 4)
opacities_init = raw_splats.opacities.to(DEVICE).detach().float()   # (N,)

# ── camera setup ──────────────────────────────────────────────────────────────
def get_camera_params(sample, device):
    gt_image = sample["image_rgb"].to(device)           # (1, 3, H, W)
    fg_mask  = sample["fg_probability"].to(device)      # (1, 1, H, W)
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

# ── 2DGS render ───────────────────────────────────────────────────────────────
def render_2dgs(means, quats, scales2d, opacities, sh0, viewmats, K, h, w):
    """
    Calls gsplat.rasterization_2dgs.

    scales2d : (N, 2)  — log-scale, exponentiated inside
    quats    : (N, 4)  — unit quaternion defining tangent plane orientation
    opacities: (N,)    — pre-sigmoid logits
    """
    out = rasterization_2dgs(
        means       = means,
        quats       = F.normalize(quats, p=2, dim=-1),
        scales      = torch.cat([                            # (N, 3): disk axes + fixed thin normal
                          torch.exp(scales2d),
                          torch.full((scales2d.shape[0], 1), 1e-4, device=scales2d.device),
                      ], dim=-1),
        opacities   = torch.sigmoid(opacities.flatten()),
        colors      = sh0.unsqueeze(1),                    # (N, 1, 3)
        viewmats    = viewmats,
        Ks          = K,
        width       = w,
        height      = h,
        sh_degree   = 0,
        near_plane  = 0.01,
        backgrounds = torch.zeros(1, 3, device=DEVICE),
        packed      = False,
        absgrad     = False,
    )
    return out[0].permute(0, 3, 1, 2)                     # (1, 3, H, W)

# ── losses ────────────────────────────────────────────────────────────────────
def l2_loss(pred, gt, mask=None):
    diff = (pred - gt) ** 2
    if mask is None:
        return diff.mean()
    binary_mask = (mask > 0.5).float()
    return (diff * binary_mask).sum() / binary_mask.sum().clamp(min=1.0)

def ssim_loss(pred, gt):
    """
    Simple single-scale SSIM (window=11).
    pred, gt: (1, 3, H, W) in [0, 1]
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    kernel_size = 11
    sigma = 1.5
    # Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=pred.device)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]                       # (11, 11)
    kernel = kernel.expand(3, 1, kernel_size, kernel_size) # (3, 1, 11, 11)

    pad = kernel_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=3)
    mu2 = F.conv2d(gt,   kernel, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sig1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=3) - mu1_sq
    sig2_sq = F.conv2d(gt   * gt,   kernel, padding=pad, groups=3) - mu2_sq
    sig12   = F.conv2d(pred * gt,   kernel, padding=pad, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))
    return 1.0 - ssim_map.mean()

def reconstruction_loss(pred, gt, mask=None, lambda_ssim=0.2):
    """
    Combined L2 + SSIM loss (standard in 3DGS/2DGS papers).
    """
    return (1.0 - lambda_ssim) * l2_loss(pred, gt, mask) + \
                  lambda_ssim  * ssim_loss(pred, gt)

# ── Mini-PointNet encoder for 2DGS ───────────────────────────────────────────
# Flat per-point input:
#   means (3) + sh0 (3) + scales2d (2) + quats (4) + opacity (1) = 13
# Delta output: 13 (same layout)

SPLAT_DIM = 13   # 2DGS parameter count per Gaussian

class MiniPointNet2DGSNetwork(nn.Module):
    """
    Mini-PointNet encoder → global max-pool → broadcast → per-point decoder.
    Predicts residual deltas over 2DGS parameters.

    Input  : means (N,3) only fed to the encoder (position-only embedding)
    Flat   : [means(3) | sh0(3) | scales2d(2) | quats(4) | opacity(1)] = 13
    Output : reconstructed 2DGS splats with delta corrections
    """
    def __init__(self, splat_dim: int = SPLAT_DIM):
        super().__init__()
        self.splat_dim = splat_dim

        # ── Encoder: position-only MLP, handles all N points ──────────────────
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
        )

        # ── Decoder: global_feat(1024) + per-point flat(13) → delta(13) ───────
        self.decoder = nn.Sequential(
            nn.Linear(1024 + splat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, splat_dim),
        )

    def encode(self, means_b: torch.Tensor) -> torch.Tensor:
        """
        means_b : (B, N, 3)
        returns : (B, 1024) global feature
        """
        per_point   = self.point_encoder(means_b)      # (B, N, 1024)
        global_feat = per_point.max(dim=1).values       # (B, 1024)
        return global_feat

    def reconstruct_splats_2dgs(
        self,
        means:     torch.Tensor,   # (N, 3)
        sh0:       torch.Tensor,   # (N, 3)
        scales2d:  torch.Tensor,   # (N, 2)
        quats:     torch.Tensor,   # (N, 4)
        opacities: torch.Tensor,   # (N,)
    ):
        """
        Returns corrected 2DGS parameters as a dict.
        All outputs have the same shape as inputs.
        """
        N = means.shape[0]

        # Encode positions of ALL N Gaussians
        global_feat = self.encode(means.unsqueeze(0))   # (1, 1024)
        global_feat = global_feat.expand(N, -1)         # (N, 1024)

        # Flat per-point feature: [means | sh0 | scales2d | quats | opacity]
        flat = torch.cat([
            means,
            sh0,
            scales2d,
            quats,
            opacities.unsqueeze(-1),
        ], dim=-1)                                       # (N, 13)

        delta = self.decoder(
            torch.cat([global_feat, flat], dim=-1)
        )                                                # (N, 13)

        # Split delta into per-parameter corrections
        d_means    = delta[:, 0:3]
        d_sh0      = delta[:, 3:6]
        d_scales2d = delta[:, 6:8]
        d_quats    = delta[:, 8:12]
        d_opacity  = delta[:, 12]

        return dict(
            means     = means     + d_means,
            sh0       = sh0       + d_sh0,
            scales2d  = scales2d  + 0.02 * d_scales2d.clamp(-1, 1),
            quats     = quats     + d_quats,              # normalised in render()
            opacities = opacities + d_opacity,
        )

# ── model + optimiser ─────────────────────────────────────────────────────────
network   = MiniPointNet2DGSNetwork().to(DEVICE)
optimizer = AdamW(network.parameters(), lr=LR)

# ── reference render: original splat rendered as 2DGS (no grad) ───────────────
v0_mats, v0_K, v0_gt_image, v0_fg_mask, v0_h, v0_w = view_data[0]

with torch.no_grad():
    ref_render = render_2dgs(
        means_init, quats_init, scales2d_init,
        opacities_init, sh0_init,
        v0_mats, v0_K, v0_h, v0_w,
    )

os.makedirs("logs_2dgs_minipointnet", exist_ok=True)

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    network.train()
    total_epoch_loss = 0.0

    for v_mats, v_K, v_gt_image, v_fg_mask, v_h, v_w in view_data:
        optimizer.zero_grad()

        recon = network.reconstruct_splats_2dgs(
            means_init, sh0_init, scales2d_init, quats_init, opacities_init,
        )

        rendered = render_2dgs(
            recon["means"],
            recon["quats"],
            recon["scales2d"],
            recon["opacities"],
            recon["sh0"],
            v_mats, v_K, v_h, v_w,
        )

        loss = reconstruction_loss(rendered, v_gt_image, mask=v_fg_mask, lambda_ssim=0.2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        total_epoch_loss += loss.item()

    # ── logging ───────────────────────────────────────────────────────────────
    if epoch % LOG_EVERY == 0:
        avg_loss = total_epoch_loss / NUM_VIEWS
        network.eval()
        with torch.no_grad():
            recon_eval = network.reconstruct_splats_2dgs(
                means_init, sh0_init, scales2d_init, quats_init, opacities_init,
            )
            eval_rendered = render_2dgs(
                recon_eval["means"],
                recon_eval["quats"],
                recon_eval["scales2d"],
                recon_eval["opacities"],
                recon_eval["sh0"],
                v0_mats, v0_K, v0_h, v0_w,
            )

        save_path = f"logs_2dgs_minipointnet/epoch_{epoch:05d}.png"
        SplatRenderVisualizer.save(
            save_path,
            render_colors = torch.cat([
                v0_gt_image.permute(0, 2, 3, 1),   # GT image
                ref_render.permute(0, 2, 3, 1),    # original splat (2DGS, no network)
                eval_rendered.permute(0, 2, 3, 1), # network reconstruction
            ], dim=2),
            render_alphas = torch.ones(1, v0_h, v0_w * 3, 1, device=DEVICE),
        )
        print(f"Epoch {epoch:05d}/{EPOCHS} | Avg Loss = {avg_loss:.6f} | Saved {save_path}")

# ── final visualisation ───────────────────────────────────────────────────────
network.eval()
with torch.no_grad():
    recon_final = network.reconstruct_splats_2dgs(
        means_init, sh0_init, scales2d_init, quats_init, opacities_init,
    )
    final_render = render_2dgs(
        recon_final["means"],
        recon_final["quats"],
        recon_final["scales2d"],
        recon_final["opacities"],
        recon_final["sh0"],
        v0_mats, v0_K, v0_h, v0_w,
    )

SplatRenderVisualizer.save(
    "gt_vs_recon_2dgs_final.png",
    render_colors = torch.cat([
        v0_gt_image.permute(0, 2, 3, 1),
        ref_render.permute(0, 2, 3, 1),
        final_render.permute(0, 2, 3, 1),
    ], dim=2),
    render_alphas = torch.ones(1, v0_h, v0_w * 3, 1, device=DEVICE),
)
print("Saved gt_vs_recon_2dgs_final.png")
