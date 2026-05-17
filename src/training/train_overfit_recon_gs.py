import torch
import os
import torch.nn.functional as F
from torch.optim import AdamW
from gsplat import rasterization
from uco3d.dataset_utils.data_types import GaussianSplats
from src.models.simple_recon_autoencoder import AbsoluteReconAutoencoder, POC_Network
from src.data.datamodule import UCO3DDataModule
from src.utils.splat_visualizer import SplatRenderVisualizer
from src.utils.rendering import get_camera_params, render
from src.utils.losses import l2_loss

os.environ["UCO3D_DATASET_ROOT"] = "/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw"

EPOCHS     = 5000
LR         = 1e-4
NUM_VIEWS  = 1
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_WEIGHTS = {
    "means":     1.0,
    "sh0":       1.0,
    "scales":    0.5,
    "opacities": 0.5,
    "quats":     0.1,
}
LAMBDA_PHOTO = 1.0   # weight for photometric loss
LAMBDA_RECON = 0.0   # weight for Gaussian parameter loss

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

view_data = [get_camera_params(f, DEVICE) for f in training_frames]

# ── gaussian parameter reconstruction loss ────────────────────────────────────
def gaussian_reconstruction_loss(pred, gt, weights=None):
    """
    Direct parameter-space loss between predicted and GT Gaussians.
    Assumes fixed 1-to-1 index correspondence between pred and gt Gaussians.

    pred, gt : GaussianSplats with .means, .sh0, .scales, .quats, .opacities
    weights  : dict of per-attribute loss weights (optional)
    """
    w = weights or {}

    # Position loss (L2)
    loss_means = F.mse_loss(pred.means, gt.means)

    # Color (SH degree-0) loss (L2)
    loss_sh0 = F.mse_loss(pred.sh0, gt.sh0)

    # Scale loss — both in log-space, compare directly
    loss_scales = F.mse_loss(pred.scales, gt.scales)

    # Opacity loss — both pre-sigmoid logits, compare directly
    loss_opacities = F.mse_loss(pred.opacities, gt.opacities)

    # Rotation loss — geodesic distance on unit quaternions
    # Handles the q ≡ -q double-cover ambiguity via abs()
    pred_q = F.normalize(pred.quats, p=2, dim=-1)
    gt_q   = F.normalize(gt.quats,   p=2, dim=-1)
    dot    = (pred_q * gt_q).sum(dim=-1).abs().clamp(max=1.0)
    loss_quats = (1.0 - dot).mean()

    total = (
        w.get("means",     1.0) * loss_means     +
        w.get("sh0",       1.0) * loss_sh0       +
        w.get("scales",    0.5) * loss_scales    +
        w.get("opacities", 0.5) * loss_opacities +
        w.get("quats",     0.1) * loss_quats
    )

    breakdown = {
        "means":     loss_means.item(),
        "sh0":       loss_sh0.item(),
        "scales":    loss_scales.item(),
        "opacities": loss_opacities.item(),
        "quats":     loss_quats.item(),
    }

    return total, breakdown

# ── model + optimiser ─────────────────────────────────────────────────────────
network = AbsoluteReconAutoencoder(input_dim=14, output_dim=14).to(DEVICE)
optimizer = AdamW(network.parameters(), lr=LR)

# ── reference render (no grad) ────────────────────────────────────────────────
v0_mats, v0_K, v0_gt_image, v0_fg_mask, v0_h, v0_w = view_data[0]

with torch.no_grad():
    ref_render = render(
        input_splats.means,
        input_splats.quats,
        input_splats.scales,
        input_splats.opacities,
        input_splats.sh0,
        v0_mats, v0_K, v0_h, v0_w, DEVICE
    )

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    network.train()
    total_epoch_loss  = 0.0
    total_photo_loss  = 0.0
    total_param_loss  = 0.0

    for v_mats, v_K, v_gt_image, v_fg_mask, v_h, v_w in view_data:
        optimizer.zero_grad()

        recon = network.reconstruct_splats(input_splats)

        # 1. Photometric loss (image space)
        rendered = render(
            recon.means,
            recon.quats,
            recon.scales,
            recon.opacities,
            recon.sh0,
            v_mats, v_K, v_h, v_w, DEVICE
        )
        photo_loss = l2_loss(rendered, v_gt_image, mask=v_fg_mask)

        # 2. Gaussian parameter reconstruction loss (parameter space)
        param_loss, param_breakdown = gaussian_reconstruction_loss(
            recon, input_splats, weights=LOSS_WEIGHTS
        )

        # Combined loss
        loss = LAMBDA_PHOTO * photo_loss + LAMBDA_RECON * param_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        total_epoch_loss += loss.item()
        total_photo_loss += photo_loss.item()
        total_param_loss += param_loss.item()

    if epoch % 100 == 0:
        avg_loss  = total_epoch_loss / NUM_VIEWS
        avg_photo = total_photo_loss / NUM_VIEWS
        avg_param = total_param_loss / NUM_VIEWS

        network.eval()
        with torch.no_grad():
            recon_eval   = network.reconstruct_splats(input_splats)
            eval_rendered = render(
                recon_eval.means,
                recon_eval.quats,
                recon_eval.scales,
                recon_eval.opacities,
                recon_eval.sh0,
                v0_mats, v0_K, v0_h, v0_w, DEVICE
            )
            SplatRenderVisualizer.save(
                "gt_vs_recon.png",
                render_colors = torch.cat([
                    v0_gt_image.permute(0, 2, 3, 1),               # GT
                    ref_render.permute(0, 2, 3, 1),                 # original splats render
                    eval_rendered.permute(0, 2, 3, 1),               # network reconstruction
                ], dim=2),
                render_alphas = torch.ones(1, v0_h, v0_w * 3, 1, device=DEVICE),
            )

        print(
            f"Epoch {epoch:05d}/{EPOCHS} | "
            f"Loss = {avg_loss:.6f} | "
            f"Photo = {avg_photo:.6f} | "
            f"Param = {avg_param:.6f}"
        )
        print(
            f"  └─ means={param_breakdown['means']:.6f}  "
            f"sh0={param_breakdown['sh0']:.6f}  "
            f"scales={param_breakdown['scales']:.6f}  "
            f"opacities={param_breakdown['opacities']:.6f}  "
            f"quats={param_breakdown['quats']:.6f}"
        )