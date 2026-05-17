import torch
import os
import torch.nn.functional as F
from torch.optim import AdamW
from torch import nn
from src.data.datamodule import UCO3DDataModule
from src.utils.splat_visualizer import SplatRenderVisualizer
from src.utils.rendering import get_camera_params, render_2dgs
from src.models.minipointnet import MiniPointNet2DGSNetwork
from src.utils.losses import reconstruction_loss

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
means_init     = raw_splats.means.to(DEVICE).detach().float()       # (N, 3)
sh0_init       = raw_splats.sh0.to(DEVICE).detach().float()         # (N, 3)
scales_init    = raw_splats.scales.to(DEVICE).detach().float()      # (N, 3)
scales2d_init  = scales_init[:, :2]                                 # (N, 2)  drop depth
quats_init     = raw_splats.quats.to(DEVICE).detach().float()       # (N, 4)
opacities_init = raw_splats.opacities.to(DEVICE).detach().float()   # (N,)

view_data = [get_camera_params(f, DEVICE) for f in training_frames]

# ── model + optimiser ─────────────────────────────────────────────────────────
network   = MiniPointNet2DGSNetwork().to(DEVICE)
optimizer = AdamW(network.parameters(), lr=LR)

# ── reference render: original splat rendered as 2DGS (no grad) ───────────────
v0_mats, v0_K, v0_gt_image, v0_fg_mask, v0_h, v0_w = view_data[0]

with torch.no_grad():
    ref_render = render_2dgs(
        means_init, quats_init, scales2d_init,
        opacities_init, sh0_init,
        v0_mats, v0_K, v0_h, v0_w, DEVICE
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
            v_mats, v_K, v_h, v_w, DEVICE
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
                v0_mats, v0_K, v0_h, v0_w, DEVICE
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