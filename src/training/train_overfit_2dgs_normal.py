from src.models.minipointnet import MiniPointNet2DGSNetwork
import torch
import os
import torch.nn.functional as F
from torch.optim import AdamW
from src.data.datamodule import UCO3DDataModule
from src.utils.splat_visualizer import SplatRenderVisualizer
from src.utils.losses import reconstruction_loss
from src.utils.rendering import render_2dgs, render, get_camera_params

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


view_data = [get_camera_params(f, DEVICE, ret_normals=True) for f in training_frames]

# ── model + optimiser ─────────────────────────────────────────────────────────
network   = MiniPointNet2DGSNetwork().to(DEVICE)
optimizer = AdamW(network.parameters(), lr=LR)

# ── reference render: clean 3DGS rasterization (no network, no grad) ──────────
v0_mats, v0_K, v0_gt_image, v0_fg_mask, v0_h, v0_w, v0_R_c2w, v0_gt_normals = view_data[0]

with torch.no_grad():
    # Clean 3DGS RGB reference
    ref_render = render(
        means_init, quats_init, scales_init,
        opacities_init, sh0_init,
        v0_mats, v0_K, v0_h, v0_w, DEVICE
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
            v_mats, v_K, v_h, v_w, DEVICE, ret_normals=True
        )

        loss_rgb    = reconstruction_loss(rgb, v_gt_image, mask=v_fg_mask)
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
                v0_mats, v0_K, v0_h, v0_w, DEVICE, ret_normals=True
            )
            eval_normals_vis = (eval_normals * 0.5 + 0.5).clamp(0, 1)

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