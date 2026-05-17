import torch
import os
from torch.optim import AdamW
from uco3d.dataset_utils.data_types import GaussianSplats
from models.simple_recon_autoencoder import ReconAutoencoder
from data.datamodule import UCO3DDataModule
from utils.splat_visualizer import SplatRenderVisualizer
from utils.rendering import get_camera_params, render
from utils.losses import l2_loss

os.environ["UCO3D_DATASET_ROOT"] = "/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw"

EPOCHS     = 10000
LR         = 1e-4
NUM_VIEWS  = 10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── data ──────────────────────────────────────────────────────────────────────
dm = UCO3DDataModule(batch_size=1, num_workers=0, image_height=512, image_width=512, load_masks=True)
torch.manual_seed(1)
train_loader = dm.train_loader()
print(f"Dataset size: {len(train_loader.dataset)}")

train_iter     = iter(train_loader)
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

view_data = [get_camera_params(f, DEVICE) for f in training_frames]                 # (1, 3, H, W)

# ── model + optimiser ─────────────────────────────────────────────────────────
network   = ReconAutoencoder(14).to(DEVICE)
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
    total_epoch_loss = 0.0

    for v_mats, v_K, v_gt_image, v_fg_mask, v_h, v_w in view_data:
        optimizer.zero_grad()

        recon = network.reconstruct_splats(input_splats)

        rendered = render(
            recon.means,
            recon.quats,
            recon.scales,
            recon.opacities,
            recon.sh0,
            v_mats, v_K, v_h, v_w, DEVICE
        )

        loss = l2_loss(rendered, v_gt_image, mask=v_fg_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        total_epoch_loss += loss.item()

    if epoch % 100 == 0:
        avg_loss = total_epoch_loss / NUM_VIEWS
        network.eval()
        with torch.no_grad():
            # Re-run reconstruction — don't reuse stale loop variable
            recon_eval = network.reconstruct_splats(input_splats)
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

        print(f"Epoch {epoch:05d}/{EPOCHS} | Avg Loss = {avg_loss:.6f}")