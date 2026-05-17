from models.minipointnet_2d import MiniPointNet2DGSNetwork
import torch
import os
import torch.nn.functional as F
from torch.optim import AdamW
from torch import nn
from gsplat import rasterization, rasterization_2dgs
from uco3d.dataset_utils.utils import opencv_cameras_projection_from_uco3d
from uco3d.dataset_utils.data_types import GaussianSplats
from src.data.datamodule import UCO3DDataModule
from src.utils.splat_visualizer import SplatRenderVisualizer

os.environ["UCO3D_DATASET_ROOT"] = "/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw"

EPOCHS         = 10000
LR             = 1e-4
NUM_VIEWS      = 10
LOG_EVERY      = 50
LAMBDA_SSIM    = 0.2    # weight of SSIM in reconstruction loss
LAMBDA_NORMAL  = 0.2   # weight of normal consistency loss
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── data ──────────────────────────────────────────────────────────────────────
dm = UCO3DDataModule(
    batch_size=1, num_workers=0,
    image_height=512, image_width=512,
    load_masks=True,
    load_depths=True,
)
torch.manual_seed(1)
train_loader    = dm.train_loader()
print(f"Dataset size: {len(train_loader.dataset)}")

train_iter      = iter(train_loader)
training_frames = [next(train_iter) for _ in range(NUM_VIEWS)]

raw_splats = training_frames[0]["sequence_gaussian_splats"][0]

# ── input splats (3DGS from uCO3D) ───────────────────────────────────────────
means_init     = raw_splats.means.to(DEVICE).detach().float()       # (N, 3)
sh0_init       = raw_splats.sh0.to(DEVICE).detach().float()         # (N, 3)
scales_init    = raw_splats.scales.to(DEVICE).detach().float()      # (N, 3)
scales2d_init  = scales_init[:, :2]                                 # (N, 2)
quats_init     = raw_splats.quats.to(DEVICE).detach().float()       # (N, 4)
opacities_init = raw_splats.opacities.to(DEVICE).detach().float()   # (N,)

# ── depth → surface normals ───────────────────────────────────────────────────
def depth_to_normals(depth: torch.Tensor, K: torch.Tensor,
                     valid: torch.Tensor) -> torch.Tensor:
    B, _, H, W = depth.shape
    device = depth.device

    fx = K[:, 0, 0].view(B, 1, 1)   # (B,1,1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)

    u = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, W)
    v = torch.arange(H, dtype=torch.float32, device=device).view(1, H, 1)

    Z = depth[:, 0]                  # (B, H, W)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    xyz = torch.stack([X, Y, Z], dim=1)   # (B, 3, H, W)

    def cdiff(t, dim):
        """Central finite diff along spatial dim (2=H, 3=W)."""
        d = torch.zeros_like(t)
        if dim == 2:
            d[:, :, 1:-1] = t[:, :, 2:] - t[:, :, :-2]
            d[:, :,  0]   = t[:, :, 1]  - t[:, :,  0]
            d[:, :, -1]   = t[:, :, -1] - t[:, :, -2]
        else:
            d[:, :, :, 1:-1] = t[:, :, :, 2:] - t[:, :, :, :-2]
            d[:, :, :,  0]   = t[:, :, :, 1]  - t[:, :, :,  0]
            d[:, :, :, -1]   = t[:, :, :, -1] - t[:, :, :, -2]
        return d

    dPdu = cdiff(xyz, dim=3)   # (B, 3, H, W)  horizontal tangent
    dPdv = cdiff(xyz, dim=2)   # (B, 3, H, W)  vertical tangent

    # cross product: (B, 3, H, W)
    normals = torch.cross(
        dPdu.permute(0, 2, 3, 1),
        dPdv.permute(0, 2, 3, 1),
        dim=-1,
    ).permute(0, 3, 1, 2)

    valid_f = (valid > 0).float()                            # (B, 1, H, W)
    normals = normals * valid_f

    norm_len = normals.norm(dim=1, keepdim=True).clamp(min=1e-6)
    normals  = (normals / norm_len) * valid_f                # unit, masked

    return normals


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
    # Camera-to-world rotation for transforming normals to world space
    # viewmats is world-to-camera: R = viewmats[:, :3, :3]
    R_c2w = viewmats[:, :3, :3].transpose(1, 2)        # (1, 3, 3)

    # Ground-truth normals from depth map
    depth_map  = sample["depth_map"].to(device)         # (1, 1, H, W)
    depth_mask = sample["depth_mask"].to(device)        # (1, 1, H, W)
    scale_adj  = sample["depth_scale_adjustment"]
    if hasattr(scale_adj, "to"):
        scale_adj = scale_adj.to(device).view(-1, 1, 1, 1).float()
    else:
        scale_adj = float(scale_adj)
    depth_map  = depth_map * scale_adj                  # apply metric scale
    gt_normals_depth = depth_to_normals(depth_map, K, depth_mask)  # (1,3,H,W)

    return viewmats, K, gt_image, fg_mask, h, w, R_c2w, gt_normals_depth

view_data = [get_camera_params(f, DEVICE) for f in training_frames]

# ── 3DGS reference render (clean, using proper rasterization) ─────────────────
def render_3dgs_ref(means, quats, scales, opacities, sh0, viewmats, K, h, w):
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
    return imgs.permute(0, 3, 1, 2)                     # (1, 3, H, W)

# ── 2DGS render — returns RGB + rendered normals ──────────────────────────────
def render_2dgs(means, quats, scales2d, opacities, sh0, viewmats, K, h, w):
    out = rasterization_2dgs(
        means       = means,
        quats       = F.normalize(quats, p=2, dim=-1),
        scales      = torch.cat([
                          torch.exp(scales2d),
                          torch.full((scales2d.shape[0], 1), 1e-4, device=scales2d.device),
                      ], dim=-1),                        # (N, 3)
        opacities   = torch.sigmoid(opacities.flatten()),
        colors      = sh0.unsqueeze(1),                  # (N, 1, 3)
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
    # Unpack — shape varies by gsplat version, inspect at runtime
    rgb = out[0].permute(0, 3, 1, 2)                    # (1, 3, H, W) always first

    # Find the normals tensor: first output after RGB that has 3 channels
    normals = None
    for tensor in out[1:]:
        t = tensor.permute(0, 3, 1, 2) if tensor.dim() == 4 else tensor
        if t.shape[1] == 3:
            normals = t
            break

    return rgb, normals

# ── losses ────────────────────────────────────────────────────────────────────
def l2_loss(pred, gt, mask=None):
    diff = (pred - gt) ** 2
    if mask is None:
        return diff.mean()
    binary_mask = (mask > 0.5).float()
    return (diff * binary_mask).sum() / binary_mask.sum().clamp(min=1.0)

def ssim_loss(pred, gt):
    C1, C2     = 0.01 ** 2, 0.03 ** 2
    ks, sigma  = 11, 1.5
    coords = torch.arange(ks, dtype=torch.float32, device=pred.device) - ks // 2
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g     /= g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, ks, ks)
    pad    = ks // 2

    mu1    = F.conv2d(pred,      kernel, padding=pad, groups=3)
    mu2    = F.conv2d(gt,        kernel, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1mu2 = mu1**2, mu2**2, mu1*mu2

    s1  = F.conv2d(pred*pred, kernel, padding=pad, groups=3) - mu1_sq
    s2  = F.conv2d(gt*gt,     kernel, padding=pad, groups=3) - mu2_sq
    s12 = F.conv2d(pred*gt,   kernel, padding=pad, groups=3) - mu1mu2

    ssim_map = ((2*mu1mu2 + C1)*(2*s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1)*(s1 + s2 + C2))
    return 1.0 - ssim_map.mean()

def reconstruction_loss(pred, gt, mask=None):
    return (1.0 - LAMBDA_SSIM) * l2_loss(pred, gt, mask) + \
                  LAMBDA_SSIM  * ssim_loss(pred, gt)

def normal_consistency_loss(normals, mask=None):
    # ── 1. Smoothness via finite differences ──────────────────────────────────
    dx = normals[:, :, :, 1:] - normals[:, :, :, :-1]  # (1,3,H,W-1)
    dy = normals[:, :, 1:, :] - normals[:, :, :-1, :]  # (1,3,H-1,W)

    if mask is not None:
        binary = (mask > 0.5).float()
        dx = dx * binary[:, :, :, 1:]
        dy = dy * binary[:, :, 1:, :]

    smooth = dx.abs().mean() + dy.abs().mean()

    # ── 2. Unit-norm penalty ───────────────────────────────────────────────────
    norms     = normals.norm(dim=1, keepdim=True)           # (1,1,H,W)
    unit_pen  = (norms - 1.0).abs()
    if mask is not None:
        unit_pen = unit_pen * (mask > 0.5).float()
    unit_pen  = unit_pen.mean()

    return smooth + unit_pen

# ── model + optimiser ─────────────────────────────────────────────────────────
network   = MiniPointNet2DGSNetwork().to(DEVICE)
optimizer = AdamW(network.parameters(), lr=LR)

# ── reference render: clean 3DGS rasterization (no network, no grad) ──────────
# ── reference renders (no grad) ───────────────────────────────────────────────
v0_mats, v0_K, v0_gt_image, v0_fg_mask, v0_h, v0_w, v0_R_c2w, v0_gt_normals = view_data[0]

with torch.no_grad():
    # Clean 3DGS RGB reference
    ref_render = render_3dgs_ref(
        means_init, quats_init, scales_init,
        opacities_init, sh0_init,
        v0_mats, v0_K, v0_h, v0_w,
    )

# GT normals come from depth maps (one per view, already on device)
gt_normals_vis = (v0_gt_normals * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1] for vis

os.makedirs("logs_2dgs_minipointnet_normal", exist_ok=True)

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    network.train()
    total_loss_rgb    = 0.0
    total_loss_normal = 0.0

    for v_mats, v_K, v_gt_image, v_fg_mask, v_h, v_w, v_R_c2w, v_gt_normals in view_data:
        optimizer.zero_grad()

        recon = network.reconstruct_splats_2dgs(
            means_init, sh0_init, scales2d_init, quats_init, opacities_init,
        )

        rgb, normals = render_2dgs(
            recon["means"], recon["quats"], recon["scales2d"],
            recon["opacities"], recon["sh0"],
            v_mats, v_K, v_h, v_w,
        )

        loss_rgb    = reconstruction_loss(rgb, v_gt_image, mask=v_fg_mask)
        # Use depth-derived gt normals: L1 where both depth and fg mask are valid
        depth_valid = (v_gt_normals.norm(dim=1, keepdim=True) > 0.5).float()
        normal_mask = (v_fg_mask > 0.5).float() * depth_valid
        loss_normal = F.l1_loss(normals * normal_mask, v_gt_normals * normal_mask)
        loss        = loss_rgb + LAMBDA_NORMAL * loss_normal

        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_rgb    += loss_rgb.item()
        total_loss_normal += loss_normal.item()

    # ── logging ───────────────────────────────────────────────────────────────
    if epoch % LOG_EVERY == 0:
        avg_rgb    = total_loss_rgb    / NUM_VIEWS
        avg_normal = total_loss_normal / NUM_VIEWS

        network.eval()
        with torch.no_grad():
            recon_eval = network.reconstruct_splats_2dgs(
                means_init, sh0_init, scales2d_init, quats_init, opacities_init,
            )
            eval_rgb, eval_normals = render_2dgs(
                recon_eval["means"], recon_eval["quats"], recon_eval["scales2d"],
                recon_eval["opacities"], recon_eval["sh0"],
                v0_mats, v0_K, v0_h, v0_w,
            )
            eval_normals_vis = (eval_normals * 0.5 + 0.5).clamp(0, 1)

        # Save: GT RGB | 3DGS ref | 2DGS RGB | GT normals | pred normals
        save_path = f"logs_2dgs_minipointnet_normal/epoch_{epoch:05d}.png"
        SplatRenderVisualizer.save(
            save_path,
            render_colors = torch.cat([
                v0_gt_image.permute(0, 2, 3, 1),        # GT RGB
                ref_render.permute(0, 2, 3, 1),          # 3DGS reference RGB
                eval_rgb.permute(0, 2, 3, 1),            # 2DGS predicted RGB
                gt_normals_vis.permute(0, 2, 3, 1),      # GT normals (from input splat)
                eval_normals_vis.permute(0, 2, 3, 1),    # predicted normals
            ], dim=2),
            render_alphas = torch.ones(1, v0_h, v0_w * 5, 1, device=DEVICE),
        )
        print(
            f"Epoch {epoch:05d}/{EPOCHS} | "
            f"RGB={avg_rgb:.6f} | Normal={avg_normal:.6f} | "
            f"Saved {save_path}"
        )