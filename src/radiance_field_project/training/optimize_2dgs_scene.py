"""optimize_2dgs_scene.py
================================
Direct 2D Gaussian Splatting optimisation for a single scene.

What this script does
---------------------
1. Loads one UCO3D sequence with multi-view RGB images, depth maps, and
   the pre-reconstructed 3DGS (Gaussian Splats) from uCO3D.
2. Converts the 3DGS to 2DGS canonical form (thin axis → normal direction)
   using ``to_2dgs_canonical``.
3. Directly optimises all 2DGS parameters as leaf tensors (no network),
   supervised by a per-view RGB + normal-consistency loss.
4. After training, saves a two-row × three-column matplotlib figure:

       Row 0 (RGB):     GT RGB      |  3D-GS init RGB    |  Optimised 2DGS RGB
       Row 1 (normals): GT normals  |  3D-GS init normals|  Optimised 2DGS normals

Run
---
    cd /home/nrieger/radiance-field-project
    uv run python src/radiance_field_project/training/optimize_2dgs_scene.py
"""

from __future__ import annotations
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Must be set before any uco3d / radiance_field_project imports so that
# paths.py picks up the correct root at module-load time.
os.environ["UCO3D_DATASET_ROOT"] = (
    "/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw"
)

from radiance_field_project.data.datamodule import UCO3DDataModule     # noqa: E402
from radiance_field_project.utils.rendering import (                    # noqa: E402
    render_2dgs, render, get_camera_params,
)
from radiance_field_project.utils.gaussian_ops import to_2dgs_canonical # noqa: E402
from radiance_field_project.utils.losses import reconstruction_loss      # noqa: E402
from uco3d.dataset_utils.data_types import Cameras                       # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS        = 20000      # direct-optimisation converges fast
LR            = 3e-4
NUM_VIEWS     = 10        # views sampled evenly from the sequence
MAX_GAUSSIANS = 50_000    # subsample Gaussians for speed; None = use all
LOG_EVERY     = 100
LAMBDA_NORMAL = 0.2       # normal-consistency loss weight
IMAGE_SIZE    = 256       # render resolution
OUT_DIR       = "logs_2dgs_direct"
FINAL_PLOT    = os.path.join(OUT_DIR, "normals_comparison_final.png")


# ─────────────────────────────────────────────────────────────────────────────
# Frame-slicing helpers  (UCO3D batches all frames of a sequence together)
# ─────────────────────────────────────────────────────────────────────────────
def _slice_camera(cam: Cameras, i: int) -> Cameras:
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


def slice_frame(batch: dict, i: int) -> dict:
    """Return a single-frame dict from a batched sequence dict."""
    dsa = batch["depth_scale_adjustment"]
    return {
        "image_rgb":              batch["image_rgb"][i : i + 1],
        "fg_probability":         batch["fg_probability"][i : i + 1],
        "camera":                 _slice_camera(batch["camera"], i),
        "depth_map":              batch["depth_map"][i : i + 1],
        "depth_mask":             batch["depth_mask"][i : i + 1],
        "depth_scale_adjustment": dsa[i : i + 1] if torch.is_tensor(dsa) else dsa,
        "sequence_gaussian_splats": batch["sequence_gaussian_splats"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def chw_to_hwc(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) tensor → (H,W,3) numpy float32 in [0,1]."""
    return t[0].permute(1, 2, 0).detach().cpu().float().numpy().clip(0, 1)


def normals_to_hwc(n: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) normal map in [-1,1] → (H,W,3) RGB in [0,1]."""
    return chw_to_hwc((n * 0.5 + 0.5).clamp(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ─────────────────────────────────────────────────────────────────
    # UCO3D DataLoader: SequenceBatchSampler groups ALL frames of one sequence
    # into a single batch → one iteration yields the whole sequence at once.
    print("Loading dataset (one sequence) …")
    dm = UCO3DDataModule(
        num_workers=0,
        image_height=IMAGE_SIZE,
        image_width=IMAGE_SIZE,
        load_masks=True,
        load_depths=True,
        # single_sequence=True (default) → picks sequence '26-14460-30571'
    )
    torch.manual_seed(42)
    train_loader = dm.train_loader()

    # Grab the one batch that contains all frames of the sequence
    seq_batch = next(iter(train_loader))   # dict with tensors of shape (T, …)
    T = seq_batch["image_rgb"].shape[0]    # total frames in sequence
    print(f"Sequence has {T} frames")

    # Pick NUM_VIEWS evenly-spaced frame indices
    step        = max(1, T // NUM_VIEWS)
    frame_idxs  = list(range(0, T, step))[:NUM_VIEWS]
    print(f"Using {len(frame_idxs)} training views: {frame_idxs}")

    # ── extract 3DGS splats ───────────────────────────────────────────────────
    raw_splats     = seq_batch["sequence_gaussian_splats"][0]
    means_full     = raw_splats.means.float()
    sh0_full       = raw_splats.sh0.float()
    scales3d_full  = raw_splats.scales.float()      # log-scales
    quats_full     = raw_splats.quats.float()
    opacities_full = raw_splats.opacities.float()
    N_full         = means_full.shape[0]
    print(f"Loaded {N_full:,} Gaussians")

    # Optional subsampling
    if MAX_GAUSSIANS is not None and N_full > MAX_GAUSSIANS:
        perm = torch.randperm(N_full)[:MAX_GAUSSIANS]
        means_init     = means_full[perm].to(device)
        sh0_init       = sh0_full[perm].to(device)
        scales3d_init  = scales3d_full[perm].to(device)
        quats_init     = quats_full[perm].to(device)
        opacities_init = opacities_full[perm].to(device)
        print(f"Subsampled to {MAX_GAUSSIANS:,} Gaussians")
    else:
        means_init     = means_full.to(device)
        sh0_init       = sh0_full.to(device)
        scales3d_init  = scales3d_full.to(device)
        quats_init     = quats_full.to(device)
        opacities_init = opacities_full.to(device)

    # ── convert 3DGS → 2DGS canonical ────────────────────────────────────────
    scales2d_init, quats2d_init = to_2dgs_canonical(scales3d_init, quats_init)

    # ── camera params (one entry per training view) ───────────────────────────
    print("Extracting camera parameters …")
    view_data = [
        get_camera_params(slice_frame(seq_batch, i), device, ret_normals=True)
        for i in frame_idxs
    ]
    # Each entry: (viewmats, K, gt_image, fg_mask, h, w, R_c2w, gt_normals)

    v0_mats, v0_K, v0_gt_image, v0_fg_mask, v0_h, v0_w, _, v0_gt_normals = \
        view_data[0]

    # ── pre-render: 3DGS-init normals (reference, no grad) ───────────────────
    # Rasterise the ORIGINAL uCO3D 3DGS as 2DGS to get their baseline normals.
    print("Pre-rendering 3DGS-init normals …")
    with torch.no_grad():
        ref_3dgs_rgb, ref_3dgs_normals = render_2dgs(
            means_init, quats2d_init, scales2d_init,
            opacities_init, sh0_init,
            v0_mats, v0_K, v0_h, v0_w, device, ret_normals=True,
        )

    # ── leaf tensors to optimise ──────────────────────────────────────────────
    means_opt     = means_init.clone().requires_grad_(True)
    sh0_opt       = sh0_init.clone().requires_grad_(True)
    scales2d_opt  = scales2d_init.clone().requires_grad_(True)
    quats_opt     = quats2d_init.clone().requires_grad_(True)
    opacities_opt = opacities_init.clone().requires_grad_(True)

    params    = [means_opt, sh0_opt, scales2d_opt, quats_opt, opacities_opt]
    optimizer = AdamW(params, lr=LR)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── training loop ─────────────────────────────────────────────────────────
    N_views = len(view_data)
    print(f"\nOptimising: {EPOCHS} epochs × {N_views} views …\n")

    for epoch in range(EPOCHS):
        total_rgb    = 0.0
        total_normal = 0.0

        for v_mats, v_K, v_gt_image, v_fg_mask, v_h, v_w, _, v_gt_normals in view_data:
            optimizer.zero_grad()

            rgb, normals = render_2dgs(
                means_opt, quats_opt, scales2d_opt,
                opacities_opt, sh0_opt,
                v_mats, v_K, v_h, v_w, device, ret_normals=True,
            )

            loss_rgb    = reconstruction_loss(rgb, v_gt_image, mask=v_fg_mask)
            depth_valid = (v_gt_normals.norm(dim=1, keepdim=True) > 0.5).float()
            normal_mask = (v_fg_mask > 0.5).float() * depth_valid
            loss_normal = F.l1_loss(
                normals * normal_mask, v_gt_normals * normal_mask
            )
            loss = loss_rgb + LAMBDA_NORMAL * loss_normal

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_rgb    += loss_rgb.item()
            total_normal += loss_normal.item()

        if epoch % LOG_EVERY == 0:
            print(
                f"Epoch {epoch:05d}/{EPOCHS}  |  "
                f"RGB={total_rgb/N_views:.5f}  |  "
                f"Normal={total_normal/N_views:.5f}"
            )
            sys.stdout.flush()

    # ── final evaluation render ───────────────────────────────────────────────
    print("\nFinal eval render …")
    with torch.no_grad():
        final_rgb, final_normals = render_2dgs(
            means_opt, quats_opt, scales2d_opt,
            opacities_opt, sh0_opt,
            v0_mats, v0_K, v0_h, v0_w, device, ret_normals=True,
        )

    # ── final plot ────────────────────────────────────────────────────────────
    # Two rows × three columns
    #   Row 0 (RGB):     GT  |  3DGS init  |  Optimised 2DGS
    #   Row 1 (Normals): GT  |  3DGS init  |  Optimised 2DGS
    print("Saving final normals comparison plot …")

    row_rgb = [
        chw_to_hwc(v0_gt_image),
        chw_to_hwc(ref_3dgs_rgb),
        chw_to_hwc(final_rgb),
    ]
    row_normals = [
        normals_to_hwc(v0_gt_normals),
        normals_to_hwc(ref_3dgs_normals),
        normals_to_hwc(final_normals),
    ]
    col_titles = [
        "GT",
        "3D-GS init\n(as 2DGS)",
        f"Optimised 2DGS\n({EPOCHS} epochs)",
    ]

    N_used = min(MAX_GAUSSIANS or N_full, N_full)
    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        figsize=(14, 10),
        gridspec_kw={"hspace": 0.06, "wspace": 0.03},
    )

    for col_idx, (rgb_img, nrm_img, ctitle) in enumerate(
        zip(row_rgb, row_normals, col_titles)
    ):
        axes[0, col_idx].imshow(rgb_img)
        axes[0, col_idx].set_title(ctitle, fontsize=12, fontweight="bold", pad=7)
        axes[0, col_idx].axis("off")

        axes[1, col_idx].imshow(nrm_img)
        axes[1, col_idx].axis("off")

    # Row labels on the left
    for row_idx, rlabel in enumerate(["RGB", "Normals"]):
        axes[row_idx, 0].set_ylabel(
            rlabel, fontsize=12, fontweight="bold", rotation=90, labelpad=6,
        )
        axes[row_idx, 0].tick_params(left=False, labelleft=False)

    fig.suptitle(
        f"2DGS Scene Optimisation  ·  {EPOCHS} epochs  ·  "
        f"{N_used:,} / {N_full:,} Gaussians\n"
        f"(seq 26-14460-30571  |  lr={LR}  |  λ_normal={LAMBDA_NORMAL}  "
        f"|  {IMAGE_SIZE}×{IMAGE_SIZE})",
        fontsize=12, y=0.998,
    )

    plt.savefig(FINAL_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓  Saved → {FINAL_PLOT}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
