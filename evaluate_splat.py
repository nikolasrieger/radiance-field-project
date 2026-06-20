#!/usr/bin/env python3
"""Evaluate a trained 3DGS→2DGS model on one random sequence.

Picks a random sequence from the dataset, runs the trained model, and saves a
4×3 grid PNG comparing the input 3DGS splat and the predicted 2DGS splat side
by side for two camera views.  Each row shows [RGB | Normals | Depth].

  Row 1: 3DGS (View 1)    — depth-based normals
  Row 2: 2DGS pred (V 1)  — rasterizer normals
  Row 3: 3DGS (View 2)
  Row 4: 2DGS pred (V 2)

Also computes per-frame losses over up to --n-eval views (same losses as
training: photometric L2+SSIM, depth L1, normal cosine/L1) and prints a
summary table.

Usage
-----
    cd radiance-field-project
    uv run python evaluate_splat.py --run runs/minipointnet_2dgs_d_n_full_0.2_maxp
    uv run python evaluate_splat.py --run runs/pointnet2_2dgs_d_n_full_0.2_maxp --seed 7
    uv run python evaluate_splat.py --run runs/... --ckpt ckpt_00500.pt --out out.png
    uv run python evaluate_splat.py --run runs/... --n-eval 20
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from uco3d.dataset_utils.data_types import Cameras

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from radiance_field_project.config import load_config
from radiance_field_project.config.schema import ModelConfig, ModelType, NormalLossType, NormalRegion
from radiance_field_project.data.collate import uco3d_collate
from radiance_field_project.data.datamodule import UCO3DDataModule, SequenceBatchSampler
from radiance_field_project.data.dataset import make_uco3d_dataset
from radiance_field_project.models.model_factory import build_model
from radiance_field_project.utils.gaussian_ops import to_2dgs_canonical
from radiance_field_project.utils.rendering import (
    get_camera_params,
    render_view,
    world_normals_to_camera,
    depth_to_normals,
)
from radiance_field_project.utils.splat_visualizer import SplatRenderVisualizer
from radiance_field_project.training.train import (
    _cache_splat,
    _to_device,
    _subsample,
    _depth_loss,
    _photometric_loss,
)


# ── camera helpers (mirrors train.py) ────────────────────────────────────────

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


def _frame_dict(batch, i: int) -> dict:
    """Extract frame i from a collated batch as a dict for get_camera_params."""
    return {
        "image_rgb":              batch["image_rgb"][i : i + 1],
        "fg_probability":         batch["fg_probability"][i : i + 1],
        "camera":                 _slice_camera(batch["camera"], i),
        "depth_scale_adjustment": 1.0,  # not used; we don't call ret_normals=True
    }


def _frame_dict_with_depth(batch, i: int) -> dict:
    """Like _frame_dict but includes depth map/mask and depth_scale_adjustment."""
    dsa = batch["depth_scale_adjustment"]
    frame = {
        "image_rgb":              batch["image_rgb"][i : i + 1],
        "fg_probability":         batch["fg_probability"][i : i + 1],
        "camera":                 _slice_camera(batch["camera"], i),
        "depth_scale_adjustment": dsa[i : i + 1] if torch.is_tensor(dsa) else dsa,
    }
    if "depth_map" in batch:
        frame["depth_map"]  = batch["depth_map"][i : i + 1]
        frame["depth_mask"] = batch["depth_mask"][i : i + 1]
    return frame


# ── visualisation helpers ─────────────────────────────────────────────────────

def _vis_normals(normals: torch.Tensor) -> torch.Tensor:
    """(1,3,H,W) camera-frame normals → (1,H,W,3) in [0,1]."""
    return (normals * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)


def _vis_depth(depth: torch.Tensor) -> torch.Tensor:
    """(1,1,H,W) expected depth → (1,H,W,3) normalised to [0,1]."""
    d = depth.squeeze(1)          # (1, H, W)
    valid = d > 0
    lo = d[valid].min() if valid.any() else d.min()
    hi = d.max()
    normed = ((d - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return normed.unsqueeze(-1).expand(-1, -1, -1, 3)


# ── checkpoint loading ────────────────────────────────────────────────────────

def _build_model_from_state(model_cfg: ModelConfig, state: dict):
    """Build a model whose architecture matches the checkpoint state dict.

    Handles old checkpoints trained with xyz-only encoder input (3 channels)
    before per-splat features were added to the SA stack.
    """
    encoder_in_channel = None
    splat_dim = None
    encoder_in_dim = None

    if model_cfg.model_type == ModelType.pointnet2:
        w = state.get("sa1.mlp_convs.0.weight")
        if w is not None:
            encoder_in_channel = w.shape[1]
    elif model_cfg.model_type == ModelType.minipointnet:
        enc_w = state.get("point_encoder.0.weight")
        dec_out_w = state.get("decoder.8.weight")  # output Linear: (splat_dim, 128)
        if enc_w is not None:
            encoder_in_dim = enc_w.shape[1]
        if dec_out_w is not None:
            splat_dim = dec_out_w.shape[0]

    return build_model(model_cfg,
                       encoder_in_channel=encoder_in_channel,
                       splat_dim=splat_dim,
                       encoder_in_dim=encoder_in_dim)


def _find_ckpt(run_dir: Path, ckpt_name: str | None) -> Path:
    if ckpt_name is not None:
        return run_dir / ckpt_name
    final = run_dir / "ckpt_final.pt"
    if final.exists():
        return final
    candidates = sorted(run_dir.glob("ckpt_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return candidates[-1]


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",  required=True,
                   help="Path to a training run directory (must contain config.yaml)")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint filename inside --run (default: ckpt_final.pt or latest)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for sequence selection (default: 0)")
    p.add_argument("--out",  default=None,
                   help="Output PNG path (default: <run>/eval_splat_seed<N>.png)")
    p.add_argument("--n-eval", type=int, default=10,
                   help="Number of views to average losses over (default: 10; 0 = skip)")
    p.add_argument("--all-losses", action="store_true",
                   help="Evaluate all three losses (photo, depth, normal) regardless of "
                        "training config lambdas")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir  = Path(args.run)
    ckpt     = _find_ckpt(run_dir, args.ckpt)
    out_path = Path(args.out) if args.out else \
               run_dir / f"eval_splat_seed{args.seed}.png"

    cfg = load_config(run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Run:    {run_dir.name}")
    print(f"Ckpt:   {ckpt.name}")
    print(f"Device: {device}")
    print(f"Output: {out_path}")

    # ── model ─────────────────────────────────────────────────────────────────
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model = _build_model_from_state(cfg.model, state).to(device)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:  {model.__class__.__name__}  ({n_params:,} params)")

    out_type   = cfg.model.output_type.value   # "3dgs" | "2dgs"
    is_2dgs    = out_type == "2dgs"
    max_points = int(cfg.data.max_points) if cfg.data.max_points else 0

    # ── pick a random sequence ────────────────────────────────────────────────
    max_splats = int(cfg.data.max_splats) if cfg.data.max_splats else None
    splat_dm = UCO3DDataModule(
        subset_lists_filename=str(cfg.data.subset_lists_filename),
        dataset_root=str(cfg.data.dataset_root),
        num_workers=0,
        image_height=int(cfg.data.image_height),
        image_width=int(cfg.data.image_width),
        load_gaussian_splats=True,
        load_masks=False,
        load_depths=False,
    )
    splat_loader = splat_dm.splat_only_loader(max_splats=max_splats)
    all_batches  = list(splat_loader.batch_sampler._batches)

    rng = random.Random(args.seed)
    chosen_batch = rng.choice(all_batches)
    frame_data   = splat_loader.dataset[chosen_batch[0]]
    seq_name     = frame_data.sequence_name
    raw_splat    = frame_data.sequence_gaussian_splats
    n_pts        = raw_splat.means.shape[0]
    print(f"Sequence: {seq_name}  ({n_pts:,} points)")

    # ── prepare splats ────────────────────────────────────────────────────────
    # 3DGS input for reference renders (raw 3D Gaussians).
    splat_3dgs_cpu = _cache_splat(raw_splat, output_type="3dgs")
    # Canonical input for the model (2DGS: thin axis → position 2; 3DGS: unchanged).
    splat_in_cpu   = _cache_splat(raw_splat, output_type=out_type)

    splat_3dgs_dev = _to_device(_subsample(splat_3dgs_cpu, max_points), device)
    splat_in_dev   = _to_device(_subsample(splat_in_cpu,   max_points), device)

    with torch.no_grad():
        splat_pred = model.reconstruct_splats(splat_in_dev)

    # ── load two camera views for the chosen sequence ─────────────────────────
    img_ds = make_uco3d_dataset(
        subsets=("train",),
        subset_lists_filename=str(cfg.data.subset_lists_filename),
        dataset_root=str(cfg.data.dataset_root),
        load_images=True,
        load_depths=False,
        load_masks=True,
        load_gaussian_splats=False,
        image_height=int(cfg.data.image_height),
        image_width=int(cfg.data.image_width),
        pick_sequences=[seq_name],
    )
    img_loader = DataLoader(
        img_ds,
        batch_sampler=SequenceBatchSampler(img_ds, shuffle=False),
        num_workers=0,
        collate_fn=uco3d_collate,
    )
    for batch in img_loader:
        T = batch["image_rgb"].shape[0]
        # Two views well-separated in the sequence.
        view_frames = [_frame_dict(batch, 0), _frame_dict(batch, T // 2)]
        break

    # ── render and build grid ─────────────────────────────────────────────────
    rows, labels = [], []

    for vi, frame in enumerate(view_frames, start=1):
        vmats, K, _, _, h, w = get_camera_params(frame, device)

        # --- Input 3DGS ---
        rgb_3dgs, depth_3dgs = render_view(
            splat_3dgs_dev, vmats, K, w, h,
            output_type="3dgs", device=device, ret_depth=True,
        )
        valid_mask = (depth_3dgs > 0).float()
        normals_3dgs = depth_to_normals(depth_3dgs, K, valid_mask)  # camera-frame

        # --- Predicted output ---
        if is_2dgs:
            rgb_pred, normals_pred_world = render_view(
                splat_pred, vmats, K, w, h,
                output_type="2dgs", device=device, ret_normals=True,
            )
            normals_pred = world_normals_to_camera(normals_pred_world, vmats)
            _, depth_pred = render_view(
                splat_pred, vmats, K, w, h,
                output_type="2dgs", device=device, ret_depth=True,
            )
        else:
            # 3DGS model: depth-based normals for both sides
            rgb_pred, depth_pred = render_view(
                splat_pred, vmats, K, w, h,
                output_type="3dgs", device=device, ret_depth=True,
            )
            valid_pred = (depth_pred > 0).float()
            normals_pred = depth_to_normals(depth_pred, K, valid_pred)

        # Each row: (1, H, 3*W, 3) — cat along width axis
        row_in = torch.cat([
            rgb_3dgs.permute(0, 2, 3, 1),
            _vis_normals(normals_3dgs),
            _vis_depth(depth_3dgs),
        ], dim=2)

        row_pred = torch.cat([
            rgb_pred.permute(0, 2, 3, 1),
            _vis_normals(normals_pred),
            _vis_depth(depth_pred),
        ], dim=2)

        rows  += [row_in, row_pred]
        labels += [
            f"3DGS RGB (V{vi})", f"3DGS Normals (V{vi})", f"3DGS Depth (V{vi})",
            f"{out_type.upper()} RGB (V{vi})", f"{out_type.upper()} Normals (V{vi})", f"{out_type.upper()} Depth (V{vi})",
        ]

    # Stack rows vertically: (1, 4*H, 3*W, 3)
    grid  = torch.cat(rows, dim=1)
    alpha = torch.ones(1, grid.shape[1], grid.shape[2], 1, device=device)

    SplatRenderVisualizer.save(
        str(out_path),
        render_colors=grid,
        render_alphas=alpha,
        labels=labels,
        columns=3,
    )
    print(f"Saved → {out_path}")

    # ── loss evaluation over N views ──────────────────────────────────────────
    n_eval = args.n_eval
    if n_eval <= 0:
        return

    loss_cfg   = cfg.train.loss
    use_depth  = loss_cfg.lambda_depth  > 0 or args.all_losses
    use_normal = (loss_cfg.lambda_normal > 0 or args.all_losses) and is_2dgs

    # Reload the image dataset with depths if needed.
    eval_ds = make_uco3d_dataset(
        subsets=("train",),
        subset_lists_filename=str(cfg.data.subset_lists_filename),
        dataset_root=str(cfg.data.dataset_root),
        load_images=True,
        load_depths=use_depth or use_normal,
        load_masks=True,
        load_gaussian_splats=False,
        image_height=int(cfg.data.image_height),
        image_width=int(cfg.data.image_width),
        pick_sequences=[seq_name],
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_sampler=SequenceBatchSampler(eval_ds, shuffle=False),
        num_workers=0,
        collate_fn=uco3d_collate,
    )

    total_photo = total_depth = total_normal = 0.0
    n_frames = 0

    with torch.no_grad():
        for batch in eval_loader:
            T = batch["image_rgb"].shape[0]
            indices = list(range(min(n_eval - n_frames, T)))
            for i in indices:
                frame = _frame_dict_with_depth(batch, i)
                has_depth = "depth_map" in frame

                if use_normal and has_depth:
                    vmats, K, gt_image, fg_mask, h, w, _, gt_normals = \
                        get_camera_params(frame, device, ret_normals=True)
                    rendered, rendered_normals_world = render_view(
                        splat_pred, vmats, K, w, h,
                        output_type=out_type, device=device, ret_normals=True,
                    )
                    normals_cam = world_normals_to_camera(rendered_normals_world, vmats)
                    depth_valid = (gt_normals.norm(dim=1, keepdim=True) > 0.5).float()
                    if loss_cfg.normal_region == NormalRegion.foreground:
                        normal_mask = (fg_mask > 0.5).float() * depth_valid
                    else:
                        normal_mask = depth_valid
                    if loss_cfg.normal_loss_type == NormalLossType.cosine:
                        dot = (normals_cam * gt_normals).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
                        normal_loss = ((1.0 - dot) * normal_mask).mean().item()
                    else:
                        normal_loss = F.l1_loss(
                            normals_cam * normal_mask, gt_normals * normal_mask
                        ).item()
                    total_normal += normal_loss
                else:
                    vmats, K, gt_image, fg_mask, h, w = get_camera_params(frame, device)
                    rendered, rendered_depth = render_view(
                        splat_pred, vmats, K, w, h,
                        output_type=out_type, device=device, ret_depth=True,
                    )

                photo_loss = _photometric_loss(rendered, gt_image, fg_mask, loss_cfg).item()
                total_photo += photo_loss

                if (use_depth or not use_normal) and has_depth:
                    if use_normal and has_depth:
                        _, rendered_depth = render_view(
                            splat_pred, vmats, K, w, h,
                            output_type=out_type, device=device, ret_depth=True,
                        )
                    dsa = frame["depth_scale_adjustment"]
                    if hasattr(dsa, "to"):
                        dsa = dsa.to(device).view(-1, 1, 1, 1).float()
                    gt_depth  = frame["depth_map"].to(device) * dsa
                    depth_mask = (frame["depth_mask"].to(device) > 0.5).float()
                    depth_loss = _depth_loss(rendered_depth, gt_depth, depth_mask).item()
                    total_depth += depth_loss

                n_frames += 1
                if n_frames >= n_eval:
                    break
            if n_frames >= n_eval:
                break

    avg_photo  = total_photo  / n_frames
    avg_depth  = total_depth  / n_frames
    avg_normal = total_normal / n_frames
    full_loss  = (avg_photo
                  + loss_cfg.lambda_depth  * avg_depth
                  + loss_cfg.lambda_normal * avg_normal)

    print()
    print(f"{'─'*44}")
    print(f"  Loss summary  ({n_frames} views,  seq={seq_name})")
    print(f"{'─'*44}")
    print(f"  Photo (L2+SSIM):  {avg_photo:.6f}")
    if use_depth:
        print(f"  Depth (L1):       {avg_depth:.6f}")
    if use_normal:
        ntype = loss_cfg.normal_loss_type.value if hasattr(loss_cfg.normal_loss_type, 'value') else str(loss_cfg.normal_loss_type)
        print(f"  Normal ({ntype}):   {avg_normal:.6f}")
    print(f"  Full (weighted):  {full_loss:.6f}")
    print(f"{'─'*44}")


if __name__ == "__main__":
    main()
