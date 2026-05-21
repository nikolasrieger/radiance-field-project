import os

from radiance_field_project.models.minipointnet import MiniPointNet2DGSNetwork
from radiance_field_project.data.datamodule import UCO3DDataModule
from uco3d.dataset_utils.data_types import Cameras
from radiance_field_project.utils.splat_visualizer import SplatRenderVisualizer
from radiance_field_project.utils.losses import reconstruction_loss
from radiance_field_project.utils.rendering import render_2dgs, render, get_camera_params
import torch
import torch.nn.functional as F
from torch.optim import AdamW

os.environ["UCO3D_DATASET_ROOT"] = "/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw"

# ── hyperparams ───────────────────────────────────────────────────────────────
EPOCHS        = 10_000
LR            = 1e-4
NUM_SEQUENCES = 5      # ← increase to train on more sequences
NUM_VIEWS     = 10     # views sampled per sequence (evenly spaced)
LOG_EVERY     = 20
LAMBDA_NORMAL = 0.2
MAX_POINTS    = 100_000
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR       = "logs_2dgs_multiseq_normal"

print(f"Device: {DEVICE}  |  sequences: {NUM_SEQUENCES}  |  views/seq: {NUM_VIEWS}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _slice_camera(cam: Cameras, i: int) -> Cameras:
    """Return a single-element Cameras by slicing all batch-dimension tensors."""
    img_size = cam.image_size
    if torch.is_tensor(img_size):
        img_size = img_size[i : i + 1]
    return Cameras(
        R=cam.R[i : i + 1],
        T=cam.T[i : i + 1],
        focal_length=cam.focal_length[i : i + 1],
        principal_point=cam.principal_point[i : i + 1],
        colmap_distortion_coeffs=cam.colmap_distortion_coeffs[i : i + 1],
        device=cam.device,
        in_ndc=cam.in_ndc,
        image_size=img_size,
    )


def slice_frame(batch, i: int) -> dict:
    """Extract frame i from a sequence batch as a single-frame dict."""
    dsa = batch["depth_scale_adjustment"]
    return {
        "image_rgb":              batch["image_rgb"][i : i + 1],
        "fg_probability":         batch["fg_probability"][i : i + 1],
        "camera":                 _slice_camera(batch["camera"], i),
        "depth_map":              batch["depth_map"][i : i + 1],
        "depth_mask":             batch["depth_mask"][i : i + 1],
        "depth_scale_adjustment": dsa[i : i + 1] if torch.is_tensor(dsa) else dsa,
    }


def subsample_gaussians(g: dict, max_points: int) -> dict:
    """Keep at most ``max_points`` Gaussian points, preserving aligned fields."""
    means = g["means"]
    if means.shape[0] <= max_points:
        return g

    perm = torch.randperm(means.shape[0], device=means.device)[:max_points]
    out = {k: g[k][perm] for k in ("means", "sh0", "scales2d", "quats", "opacities")}
    if "scales3d" in g:
        out["scales3d"] = g["scales3d"][perm]
    return out


# ── data module ───────────────────────────────────────────────────────────────
dm = UCO3DDataModule(
    num_workers=4,
    image_height=512,
    image_width=512,
    load_masks=True,
    load_depths=True,
    single_sequence=False,
)

# ── step 1: preload Gaussians ONCE (no images, very fast) ────────────────────
print("Preloading Gaussians into GPU cache...")
splat_loader = dm.splat_only_loader()
# Only load the FIRST frame of each sequence — splats are per-sequence, not per-frame
splat_loader.batch_sampler._batches = [
    b[:1] for b in splat_loader.batch_sampler._batches[:NUM_SEQUENCES]
]

gaussian_cache = {}   # seq_name -> dict of GPU tensors

for batch in splat_loader:
    seq_name = batch["sequence_name"][0] if isinstance(batch["sequence_name"], list) else batch["sequence_name"]
    raw = batch["sequence_gaussian_splats"][0]
    # store FULL unsubsampled splats on CPU for the baseline render
    means     = raw.means.float()
    sh0      = raw.sh0.float()
    scales    = raw.scales.float()   # 3D scales — needed for render()
    quats     = raw.quats.float()
    opacities = raw.opacities.float()

    # keep on CPU — moved to GPU only during forward
    gaussian_cache[seq_name] = {
        "means":    means,
        "sh0":      sh0,
        "scales3d": scales,          # full 3D scales — needed for the reference render
        "scales2d": scales[:, :2],
        "quats":    quats,
        "opacities": opacities,
    }
    print(f"  Cached Gaussians for '{seq_name}'  ({raw.means.shape[0]} points)")

print(f"Gaussian cache ready: {len(gaussian_cache)} sequences.\n")

# ── step 2: image-only loader — only NUM_VIEWS frames per sequence ────────────
image_loader = dm.image_only_loader()
image_loader.batch_sampler._batches = [
    b[:: max(1, len(b) // NUM_VIEWS)][:NUM_VIEWS]
    for b in image_loader.batch_sampler._batches[:NUM_SEQUENCES]
]
print(f"Image loader ready: {len(image_loader)} sequences, up to {NUM_VIEWS} frames each.\n")

# ── model + optimizer ─────────────────────────────────────────────────────────
network   = MiniPointNet2DGSNetwork().to(DEVICE)
optimizer = AdamW(network.parameters(), lr=LR)

os.makedirs(LOG_DIR, exist_ok=True)

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    network.train()
    total_rgb    = 0.0
    total_normal = 0.0
    n_steps      = 0

    for seq_idx, batch in enumerate(image_loader):
        seq_name = batch["sequence_name"][0] if isinstance(batch["sequence_name"], list) else batch["sequence_name"]

        # Gaussians: instant lookup from CPU, move to GPU just for this sequence
        g_cpu = gaussian_cache[seq_name]

        T = batch["image_rgb"].shape[0]
        step = max(1, T // NUM_VIEWS)
        frame_indices = list(range(0, T, step))[:NUM_VIEWS]

        view_data = [
            get_camera_params(slice_frame(batch, i), DEVICE, ret_normals=True)
            for i in frame_indices
        ]

        for v_mats, v_K, v_gt_image, v_fg_mask, v_h, v_w, _, v_gt_normals in view_data:
            optimizer.zero_grad(set_to_none=True)

            g = {k: v.to(DEVICE) for k, v in g_cpu.items()}
            g_sub = subsample_gaussians(g, MAX_POINTS)

            recon = network.reconstruct_splats_2dgs(
                g_sub["means"], g_sub["sh0"],
                g_sub["scales2d"], g_sub["quats"], g_sub["opacities"],
            )
            rgb, normals = render_2dgs(
                recon["means"], recon["quats"], recon["scales2d"],
                recon["opacities"], recon["sh0"],
                v_mats, v_K, v_h, v_w, DEVICE, ret_normals=True,
            )

            loss_rgb    = reconstruction_loss(rgb, v_gt_image, mask=v_fg_mask)
            depth_valid = (v_gt_normals.norm(dim=1, keepdim=True) > 0.5).float()
            normal_mask = (v_fg_mask > 0.5).float() * depth_valid
            loss_normal = F.l1_loss(normals * normal_mask, v_gt_normals * normal_mask)
            loss        = loss_rgb + LAMBDA_NORMAL * loss_normal

            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            total_rgb    += loss_rgb.item()
            total_normal += loss_normal.item()
            n_steps      += 1

            del g, g_sub, recon, rgb, normals
            torch.cuda.empty_cache()

    # ── logging ───────────────────────────────────────────────────────────────
    if epoch % LOG_EVERY == 0:
        avg_rgb    = total_rgb    / max(n_steps, 1)
        avg_normal = total_normal / max(n_steps, 1)

        network.eval()
        for si, (seq_name, g_cpu) in enumerate(gaussian_cache.items()):
            with torch.no_grad():
                g = {k: v.to(DEVICE) for k, v in g_cpu.items()}
                g_sub = subsample_gaussians(g, MAX_POINTS)

                for batch in image_loader:
                    bn = batch["sequence_name"][0] if isinstance(batch["sequence_name"], list) else batch["sequence_name"]
                    if bn != seq_name:
                        continue

                    v0_mats, v0_K, v0_gt_image, v0_fg_mask, v0_h, v0_w, _, v0_gt_normals = \
                        get_camera_params(slice_frame(batch, 0), DEVICE, ret_normals=True)

                    recon_eval = network.reconstruct_splats_2dgs(
                        g_sub["means"], g_sub["sh0"],
                        g_sub["scales2d"], g_sub["quats"], g_sub["opacities"],
                    )
                    eval_rgb, eval_normals = render_2dgs(
                        recon_eval["means"], recon_eval["quats"], recon_eval["scales2d"],
                        recon_eval["opacities"], recon_eval["sh0"],
                        v0_mats, v0_K, v0_h, v0_w, DEVICE, ret_normals=True,
                    )
                    ref_rgb = render(
                        g_sub["means"], g_sub["quats"], g_sub["scales3d"],
                        g_sub["opacities"], g_sub["sh0"],
                        v0_mats, v0_K, v0_h, v0_w, DEVICE,
                    )
                    break

            eval_normals_vis = (eval_normals * 0.5 + 0.5).clamp(0, 1)
            save_path = f"{LOG_DIR}/epoch_{epoch:05d}_seq{si:02d}.png"
            SplatRenderVisualizer.save(
                save_path,
                render_colors=torch.cat([
                    v0_gt_image.permute(0, 2, 3, 1),
                    ref_rgb.permute(0, 2, 3, 1),
                    eval_rgb.permute(0, 2, 3, 1),
                    (v0_gt_normals * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1),
                    eval_normals_vis.permute(0, 2, 3, 1),
                ], dim=2),
                render_alphas=torch.ones(1, v0_h, v0_w * 5, 1, device=DEVICE),
            )

        print(
            f"Epoch {epoch:05d}/{EPOCHS} | "
            f"RGB={avg_rgb:.6f} | Normal={avg_normal:.6f} | "
            f"Saved {len(gaussian_cache)} images to {LOG_DIR}/"
        )
