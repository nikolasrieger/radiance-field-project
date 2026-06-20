#!/usr/bin/env python3
"""Benchmark evaluation: model vs 3DGS and per-scene fitted 2DGS baselines.

For each sequence, samples N random views (same views used for fitting and
evaluation) and measures:
  - PSNR vs ground-truth RGB
  - Normal consistency (mean cosine similarity, camera-frame rendered vs GT)
  - Rendering time   (ms/frame)
  - Fit/inference time (ms/sequence):  fitting time for 2DGS@steps, model
    forward-pass time for the trained network, — for precomputed baselines.

Methods compared
----------------
  3DGS (dataset)    Raw precomputed Gaussians from UCO3D; no model inference.
  2DGS @  500 steps  Per-scene 2DGS optimised for  500 steps from SfM init.
  2DGS @ 1000 steps  Per-scene 2DGS optimised for 1000 steps from SfM init.
  2DGS @ 1500 steps  Per-scene 2DGS optimised for 1500 steps from SfM init.
  Model             Trained network prediction (output type from training cfg).

2DGS fitting initialises from the UCO3D point positions + colors (SfM origin),
with scales/orientations/opacities reset to standard starting values.  Fitted
splats are cached to --cache-dir so subsequent runs are instant.  The fit time
recorded at each step count is stored in the cache and re-reported on reload.

Usage
-----
    cd radiance-field-project
    uv run python eval_benchmark.py --run runs/my_run
    uv run python eval_benchmark.py --run runs/my_run --ckpt ckpt_00500.pt
    uv run python eval_benchmark.py --run runs/my_run --n-seqs 20 --n-views 10
    uv run python eval_benchmark.py --run runs/my_run --train 20
    uv run python eval_benchmark.py --run runs/my_run --cache-dir /tmp/2dgs_cache
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from uco3d import GaussianSplats

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from radiance_field_project.config import load_config
from radiance_field_project.config.schema import (
    ModelConfig, ModelType, NormalLossType, NormalRegion,
)
from radiance_field_project.data.collate import uco3d_collate
from radiance_field_project.data.datamodule import UCO3DDataModule, SequenceBatchSampler
from radiance_field_project.data.dataset import make_uco3d_dataset
from radiance_field_project.models.model_factory import build_model
from radiance_field_project.utils.rendering import (
    get_camera_params,
    render_view,
    world_normals_to_camera,
    depth_to_normals,
)
from radiance_field_project.training.train import (
    _cache_splat,
    _to_device,
    _subsample,
    _photometric_loss,
)

# Step counts for fitted 2DGS baselines.
STEP_COUNTS = [500, 1000, 1500]


# ── 2DGS cache helpers ────────────────────────────────────────────────────────

def _splat_cache_path(cache_dir: Path, seq: str, steps: int) -> Path:
    return cache_dir / f"{seq}_2dgs_{steps:04d}steps.pt"


def _save_2dgs_ckpt(path: Path, splats: GaussianSplats, fit_ms: float) -> None:
    torch.save({
        "means":           splats.means.cpu(),
        "sh0":             splats.sh0.cpu(),
        "log_scales":      splats.scales[:, :2].cpu(),  # (N, 2) learned log-scales
        "quats":           splats.quats.cpu(),
        "logit_opacities": splats.opacities.cpu(),
        "fit_ms":          fit_ms,
    }, path)


def _load_2dgs_ckpt(path: Path, device: torch.device) -> tuple[GaussianSplats, float]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    log_scales = data["log_scales"]            # (N, 2)
    # Pad to (N, 3) with a dummy sentinel for the flat axis; render_view ignores col 2.
    scales = torch.cat([log_scales, torch.zeros(log_scales.shape[0], 1)], dim=-1)
    splats = GaussianSplats(
        means=data["means"].to(device),
        sh0=data["sh0"].to(device),
        scales=scales.to(device),
        quats=data["quats"].to(device),
        opacities=data["logit_opacities"].to(device),
        shN=None, fg_mask=None,
    )
    return splats, float(data["fit_ms"])


# ── 2DGS per-scene fitting ────────────────────────────────────────────────────

def _fit_2dgs_scene(
    means_init: torch.Tensor,
    sh0_init: torch.Tensor,
    view_data: list[tuple],
    step_counts: list[int],
    device: torch.device,
    loss_cfg,
) -> dict[int, tuple[GaussianSplats, float]]:
    """Fit 2DGS parameters from SfM init, return checkpoints at each step count.

    Initialises from UCO3D point positions + colors (standard SfM init) with
    scales/orientations/opacities reset to small starting values.  Gradients are
    accumulated across views before each optimizer step.

    Args:
        means_init:   (N, 3) world-space point positions from UCO3D.
        sh0_init:     (N, 3) SH-0 (base color) from UCO3D.
        view_data:    list of (vmats, K, gt_image, fg_mask, gt_normals, h, w)
                      tuples — the same views used for evaluation.
        step_counts:  sorted list of steps at which to snapshot; the function
                      optimises for max(step_counts) steps total.
        device:       torch device.
        loss_cfg:     LossConfig from training run (same loss as training).

    Returns:
        dict mapping step → (GaussianSplats on device, cumulative_fit_ms).
    """
    N = means_init.shape[0]
    use_normals = loss_cfg.lambda_normal > 0

    # ── parameter initialisation ──────────────────────────────────────────────
    # Means and colors from SfM/UCO3D; everything else reset to standard values.
    means_p      = means_init.clone().to(device).requires_grad_(True)
    sh0_p        = sh0_init.clone().to(device).requires_grad_(True)
    # Small uniform disks:  exp(-3.5) ≈ 0.03 world units
    log_scales_p = torch.full((N, 2), -3.5, device=device, requires_grad=True)
    # Identity quaternion [w, x, y, z] per point
    quats_raw    = torch.zeros(N, 4, device=device)
    quats_raw[:, 0] = 1.0
    quats_p      = quats_raw.requires_grad_(True)
    # Nearly transparent:  sigmoid(-4) ≈ 0.018
    logit_ops_p  = torch.full((N,), -4.0, device=device, requires_grad=True)

    optimizer = Adam([
        {"params": [means_p],      "lr": 1e-4},
        {"params": [sh0_p],        "lr": 2.5e-3},
        {"params": [log_scales_p], "lr": 5e-3},
        {"params": [quats_p],      "lr": 1e-3},
        {"params": [logit_ops_p],  "lr": 5e-2},
    ])

    checkpoints: dict[int, tuple[GaussianSplats, float]] = {}
    n_views = len(view_data)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    for step in range(1, max(step_counts) + 1):
        optimizer.zero_grad()

        # Build current splats inline (render_view applies exp/sigmoid/normalize).
        scales = torch.cat([log_scales_p,
                             torch.zeros(N, 1, device=device)], dim=-1)
        splats_cur = GaussianSplats(
            means=means_p, sh0=sh0_p, scales=scales,
            quats=quats_p, opacities=logit_ops_p,
            shN=None, fg_mask=None,
        )

        # Accumulate gradients across views before stepping.
        for vmats, K, gt_image, fg_mask, gt_normals, h, w in view_data:
            if use_normals:
                rendered, normals_world = render_view(
                    splats_cur, vmats, K, w, h,
                    output_type="2dgs", device=device, ret_normals=True,
                )
                normals_cam = world_normals_to_camera(normals_world, vmats)
                photo = _photometric_loss(rendered, gt_image, fg_mask, loss_cfg)
                depth_valid = (gt_normals.norm(dim=1, keepdim=True) > 0.5).float()
                if loss_cfg.normal_region == NormalRegion.foreground:
                    normal_mask = (fg_mask > 0.5).float() * depth_valid
                else:
                    normal_mask = depth_valid
                if loss_cfg.normal_loss_type == NormalLossType.cosine:
                    dot = (normals_cam * gt_normals).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
                    nloss = ((1.0 - dot) * normal_mask).mean()
                else:
                    nloss = F.l1_loss(normals_cam * normal_mask, gt_normals * normal_mask)
                loss = photo + loss_cfg.lambda_normal * nloss
            else:
                rendered = render_view(
                    splats_cur, vmats, K, w, h,
                    output_type="2dgs", device=device,
                )
                loss = _photometric_loss(rendered, gt_image, fg_mask, loss_cfg)

            # Divide by n_views so the effective gradient magnitude is view-count-independent.
            (loss / n_views).backward()

        optimizer.step()

        if step in step_counts:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            with torch.no_grad():
                scales_snap = torch.cat(
                    [log_scales_p.detach().clone(),
                     torch.zeros(N, 1, device=device)], dim=-1,
                )
                splats_snap = GaussianSplats(
                    means=means_p.detach().clone(),
                    sh0=sh0_p.detach().clone(),
                    scales=scales_snap,
                    quats=quats_p.detach().clone(),
                    opacities=logit_ops_p.detach().clone(),
                    shN=None, fg_mask=None,
                )
            checkpoints[step] = (splats_snap, elapsed_ms)

    return checkpoints


def _save_comparison(
    panels: list[tuple[str, torch.Tensor]],
    out_path: Path,
) -> None:
    """Save a side-by-side comparison PNG.

    panels: list of (label, tensor) where tensor is (1, 3, H, W) in [0, 1].
    """
    label_h = 20
    imgs = []
    for label, t in panels:
        arr = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        canvas = Image.new("RGB", (img.width, img.height + label_h), (30, 30, 30))
        canvas.paste(img, (0, label_h))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        except OSError:
            font = ImageFont.load_default()
        draw.text((4, 2), label, fill=(240, 240, 240), font=font)
        imgs.append(canvas)
    W = sum(i.width for i in imgs)
    H = imgs[0].height
    out = Image.new("RGB", (W, H), (0, 0, 0))
    x = 0
    for img in imgs:
        out.paste(img, (x, 0))
        x += img.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def _dbg_mem(label: str) -> None:
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    print(f"  [MEM] {label}: allocated={alloc:.2f} GB  reserved={reserv:.2f} GB", flush=True)


def _get_or_fit_2dgs(
    seq_name: str,
    means_init: torch.Tensor,
    sh0_init: torch.Tensor,
    view_data: list[tuple],
    cache_dir: Path,
    step_counts: list[int],
    device: torch.device,
    loss_cfg,
) -> dict[int, tuple[GaussianSplats, float]]:
    """Return fitted 2DGS splats, loading from cache if available."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [s for s in step_counts
               if not _splat_cache_path(cache_dir, seq_name, s).exists()]

    if not missing:
        result = {}
        for s in step_counts:
            splats, fit_ms = _load_2dgs_ckpt(
                _splat_cache_path(cache_dir, seq_name, s), device)
            result[s] = (splats, fit_ms)
        return result

    # Fit and cache.
    print(f"fitting {max(step_counts)} steps …", end=" ", flush=True)
    checkpoints = _fit_2dgs_scene(
        means_init, sh0_init, view_data, step_counts, device, loss_cfg)

    for s, (splats, fit_ms) in checkpoints.items():
        _save_2dgs_ckpt(_splat_cache_path(cache_dir, seq_name, s), splats, fit_ms)

    return checkpoints


# ── metrics ───────────────────────────────────────────────────────────────────

def _psnr(pred: torch.Tensor, gt: torch.Tensor,
          mask: torch.Tensor | None = None) -> float:
    """PSNR (dB). pred/gt: (1,3,H,W) in [0,1]. mask: (1,1,H,W) optional fg."""
    diff = (pred - gt) ** 2
    if mask is not None:
        m = (mask > 0.5).float()
        mse = (diff * m).sum() / (m.sum().clamp(min=1.0) * 3)
    else:
        mse = diff.mean()
    return float("inf") if mse < 1e-10 else -10.0 * torch.log10(mse).item()


def _normal_cosine(normals_pred: torch.Tensor, normals_gt: torch.Tensor,
                   mask: torch.Tensor | None = None) -> float:
    """Mean cosine similarity between camera-frame normal maps. NaN if no valid pixels."""
    valid = (normals_gt.norm(dim=1, keepdim=True) > 0.5) & \
            (normals_pred.norm(dim=1, keepdim=True) > 0.1)
    if mask is not None:
        valid = valid & (mask > 0.5)
    if not valid.any():
        return float("nan")
    cos = (normals_pred * normals_gt).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    return cos[valid].mean().item()


# ── timed render ──────────────────────────────────────────────────────────────

def _timed_render(splats, vmats, K, w, h, output_type, device,
                  ret_normals=False, ret_depth=False):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    result = render_view(splats, vmats, K, w, h, output_type=output_type,
                         device=device, ret_normals=ret_normals, ret_depth=ret_depth)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - t0) * 1000.0


# ── camera / frame helpers ────────────────────────────────────────────────────

def _slice_camera(cam, i):
    from uco3d.dataset_utils.data_types import Cameras
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
    dsa = batch["depth_scale_adjustment"]
    return {
        "image_rgb":              batch["image_rgb"][i : i + 1],
        "fg_probability":         batch["fg_probability"][i : i + 1],
        "camera":                 _slice_camera(batch["camera"], i),
        "depth_scale_adjustment": dsa[i : i + 1] if torch.is_tensor(dsa) else dsa,
        "depth_map":              batch["depth_map"][i : i + 1],
        "depth_mask":             batch["depth_mask"][i : i + 1],
    }


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _build_model_from_state(model_cfg: ModelConfig, state: dict):
    encoder_in_channel = encoder_in_dim = splat_dim = None
    if model_cfg.model_type == ModelType.pointnet2:
        w = state.get("sa1.mlp_convs.0.weight")
        if w is not None:
            encoder_in_channel = w.shape[1]
    elif model_cfg.model_type == ModelType.minipointnet:
        enc_w = state.get("point_encoder.0.weight")
        dec_w = state.get("decoder.8.weight")
        if enc_w is not None:
            encoder_in_dim = enc_w.shape[1]
        if dec_w is not None:
            splat_dim = dec_w.shape[0]
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


# ── per-sequence evaluation ───────────────────────────────────────────────────

def evaluate_sequence(
    seq_name: str,
    splat_3dgs_cpu: GaussianSplats,
    model,
    out_type: str,
    max_points: int,
    n_views: int,
    rng: random.Random,
    cache_dir: Path,
    loss_cfg,
    subset_lists_filename: str,
    dataset_root: str,
    image_height: int,
    image_width: int,
    split: str,
    device: torch.device,
    vis_dir: Path | None = None,
) -> dict:
    """Evaluate one sequence, returning per-sequence mean metrics for all methods."""

    # ── load images + depths ──────────────────────────────────────────────────
    img_ds = make_uco3d_dataset(
        subsets=(split,),
        subset_lists_filename=subset_lists_filename,
        dataset_root=dataset_root,
        load_images=True,
        load_depths=True,
        load_masks=True,
        load_gaussian_splats=False,
        image_height=image_height,
        image_width=image_width,
        pick_sequences=[seq_name],
    )
    img_loader = DataLoader(
        img_ds,
        batch_sampler=SequenceBatchSampler(img_ds, shuffle=False),
        num_workers=0,
        collate_fn=uco3d_collate,
    )
    batch = next(iter(img_loader))
    T = batch["image_rgb"].shape[0]

    all_indices = list(range(T))
    chosen = all_indices if len(all_indices) <= n_views else sorted(rng.sample(all_indices, n_views))

    # ── precompute view data once (reused by fitting and evaluation) ──────────
    view_data: list[tuple] = []
    for i in chosen:
        frame = _frame_dict(batch, i)
        vmats, K, gt_image, fg_mask, h, w, _, gt_normals = \
            get_camera_params(frame, device, ret_normals=True)
        view_data.append((vmats, K, gt_image, fg_mask, gt_normals, h, w))

    # ── get / fit 2DGS baselines ──────────────────────────────────────────────
    # Initialise from SfM positions + colors; no max_points cap — let 2DGS use all.
    means_init = splat_3dgs_cpu.means   # (N, 3) CPU
    sh0_init   = splat_3dgs_cpu.sh0     # (N, 3) CPU

    if device.type == "cuda":
        _dbg_mem("before 2DGS fit")

    fitted_2dgs = _get_or_fit_2dgs(
        seq_name=seq_name,
        means_init=means_init,
        sh0_init=sh0_init,
        view_data=view_data,
        cache_dir=cache_dir,
        step_counts=STEP_COUNTS,
        device=device,
        loss_cfg=loss_cfg,
    )

    if device.type == "cuda":
        _dbg_mem("after 2DGS fit")

    # ── model inference (timed, no grad) ──────────────────────────────────────
    # splat_3dgs_cpu was produced by _cache_splat(raw, "3dgs"), so its scale
    # format is identical to the raw UCO3D splat.  Calling _cache_splat again
    # with out_type applies the canonical 2DGS/linprim conversion if needed.
    splat_3dgs_dev  = _to_device(_subsample(splat_3dgs_cpu, max_points), device)
    splat_model_in  = _cache_splat(splat_3dgs_cpu, out_type)
    splat_model_dev = _to_device(_subsample(splat_model_in, max_points), device)

    if device.type == "cuda":
        _dbg_mem(f"before model.reconstruct_splats  (splat_model_dev N={splat_model_dev.means.shape[0]})")

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        splat_pred = model.reconstruct_splats(splat_model_dev)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_ms = (time.perf_counter() - t0) * 1000.0

    if device.type == "cuda":
        _dbg_mem("after model.reconstruct_splats")

    # ── per-frame metrics ─────────────────────────────────────────────────────
    # Accumulators indexed as [3dgs, 2dgs@500, 2dgs@1000, 2dgs@1500, model]
    n_methods    = 1 + len(STEP_COUNTS) + 1
    psnr_sums    = [0.0] * n_methods
    normal_sums  = [0.0] * n_methods
    normal_n     = [0]   * n_methods
    render_ms    = [0.0] * n_methods

    vis_saved = False
    with torch.no_grad():
        for vmats, K, gt_image, fg_mask, gt_normals, h, w in view_data:

            # 3DGS baseline
            (rgb_3dgs, depth_3dgs), t3 = _timed_render(
                splat_3dgs_dev, vmats, K, w, h, "3dgs", device, ret_depth=True)
            normals_3dgs = depth_to_normals(depth_3dgs, K, (depth_3dgs > 0).float())
            psnr_sums[0]  += _psnr(rgb_3dgs, gt_image, fg_mask)
            cos = _normal_cosine(normals_3dgs, gt_normals, fg_mask)
            if cos == cos:
                normal_sums[0] += cos; normal_n[0] += 1
            render_ms[0] += t3

            # Fitted 2DGS baselines
            rgb_fits = {}
            for mi, steps in enumerate(STEP_COUNTS, start=1):
                splats_fit, _ = fitted_2dgs[steps]
                (rgb_fit, normals_fit_world), tf = _timed_render(
                    splats_fit, vmats, K, w, h, "2dgs", device, ret_normals=True)
                normals_fit = world_normals_to_camera(normals_fit_world, vmats)
                psnr_sums[mi]  += _psnr(rgb_fit, gt_image, fg_mask)
                cos = _normal_cosine(normals_fit, gt_normals, fg_mask)
                if cos == cos:
                    normal_sums[mi] += cos; normal_n[mi] += 1
                render_ms[mi] += tf
                rgb_fits[steps] = rgb_fit

            # Model
            mi_m = n_methods - 1
            if out_type == "2dgs":
                (rgb_m, normals_m_world), tm = _timed_render(
                    splat_pred, vmats, K, w, h, "2dgs", device, ret_normals=True)
                normals_m = world_normals_to_camera(normals_m_world, vmats)
            else:
                (rgb_m, depth_m), tm = _timed_render(
                    splat_pred, vmats, K, w, h, out_type, device, ret_depth=True)
                normals_m = depth_to_normals(depth_m, K, (depth_m > 0).float())
            psnr_sums[mi_m]  += _psnr(rgb_m, gt_image, fg_mask)
            cos = _normal_cosine(normals_m, gt_normals, fg_mask)
            if cos == cos:
                normal_sums[mi_m] += cos; normal_n[mi_m] += 1
            render_ms[mi_m] += tm

            if vis_dir is not None and not vis_saved:
                panels = [("GT", gt_image), ("3DGS", rgb_3dgs)]
                for steps in STEP_COUNTS:
                    panels.append((f"2DGS@{steps}", rgb_fits[steps]))
                panels.append(("Model", rgb_m))
                _save_comparison(panels, vis_dir / f"{seq_name}.png")
                vis_saved = True

    n = max(len(view_data), 1)
    result: dict = {"seq": seq_name, "n_frames": len(view_data), "model_ms": model_ms}

    def _norm(i): return normal_sums[i] / max(normal_n[i], 1)

    result["psnr_3dgs"]      = psnr_sums[0]     / n
    result["normal_3dgs"]    = _norm(0)
    result["render_ms_3dgs"] = render_ms[0]      / n

    for mi, steps in enumerate(STEP_COUNTS, start=1):
        _, fit_ms = fitted_2dgs[steps]
        result[f"psnr_2dgs_{steps}"]      = psnr_sums[mi]  / n
        result[f"normal_2dgs_{steps}"]    = _norm(mi)
        result[f"render_ms_2dgs_{steps}"] = render_ms[mi]  / n
        result[f"fit_ms_2dgs_{steps}"]    = fit_ms

    mi_m = n_methods - 1
    result["psnr_model"]      = psnr_sums[mi_m]  / n
    result["normal_model"]    = _norm(mi_m)
    result["render_ms_model"] = render_ms[mi_m]   / n

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",       required=True,
                   help="Training run directory (must contain config.yaml + checkpoint)")
    p.add_argument("--ckpt",      default=None,
                   help="Checkpoint filename inside --run (default: ckpt_final.pt or latest)")
    p.add_argument("--n-views",   type=int, default=10,
                   help="Random views per sequence (default: 10)")
    p.add_argument("--n-seqs",    type=int, default=0,
                   help="Max sequences to evaluate (default: 0 = all)")
    p.add_argument("--split",     default="val", choices=["train", "val", "test"],
                   help="Dataset split to evaluate on (default: val)")
    p.add_argument("--train",     type=int, default=None, metavar="N",
                   help="Shorthand for --split train --n-seqs N")
    p.add_argument("--cache-dir", default="2dgs_cache",
                   help="Directory for cached fitted 2DGS splats (default: ./2dgs_cache)")
    p.add_argument("--seed",      type=int, default=42,
                   help="RNG seed for view sampling (default: 42)")
    p.add_argument("--out-csv",   default=None,
                   help="Optional path to write per-sequence results as CSV")
    p.add_argument("--vis-dir",   default=None,
                   help="Directory to save per-sequence side-by-side comparison PNGs")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.train is not None:
        args.split = "train"
        if args.n_seqs == 0:
            args.n_seqs = args.train

    run_dir   = Path(args.run)
    ckpt      = _find_ckpt(run_dir, args.ckpt)
    cache_dir = Path(args.cache_dir)

    cfg    = load_config(run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Run:       {run_dir.name}")
    print(f"Ckpt:      {ckpt.name}")
    print(f"Split:     {args.split}")
    print(f"Cache dir: {cache_dir}")
    print(f"Device:    {device}")

    # ── model ─────────────────────────────────────────────────────────────────
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model = _build_model_from_state(cfg.model, state).to(device)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    out_type   = cfg.model.output_type.value
    max_points = int(cfg.data.max_points) if cfg.data.max_points else 0
    print(f"Model:     {model.__class__.__name__}  ({n_params:,} params, output={out_type})\n")

    # ── build splat cache ─────────────────────────────────────────────────────
    max_splats = int(cfg.data.max_splats) if cfg.data.max_splats else None
    splat_dm   = UCO3DDataModule(
        subset_lists_filename=str(cfg.data.subset_lists_filename),
        dataset_root=str(cfg.data.dataset_root),
        num_workers=0,
        image_height=int(cfg.data.image_height),
        image_width=int(cfg.data.image_width),
        load_gaussian_splats=True,
        load_masks=False,
        load_depths=False,
    )
    splat_loader = splat_dm.splat_only_loader(max_splats=max_splats, subsets=(args.split,))
    batches      = list(splat_loader.batch_sampler._batches)

    if not batches:
        print(f"No sequences found in split '{args.split}'. "
              "Try --split train if the val split is empty.")
        return

    n_seqs = len(batches) if args.n_seqs <= 0 else min(args.n_seqs, len(batches))
    print(f"Evaluating {n_seqs} sequences × {args.n_views} views "
          f"(2DGS fit steps: {STEP_COUNTS})\n")

    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"Vis dir:   {vis_dir}\n")

    rng = random.Random(args.seed)

    splat_cache: list[tuple[str, GaussianSplats]] = []
    for b in batches[:n_seqs]:
        fd  = splat_loader.dataset[b[0]]
        seq = fd.sequence_name
        raw = fd.sequence_gaussian_splats
        splat_cache.append((seq, _cache_splat(raw, "3dgs")))

    # ── per-sequence loop ──────────────────────────────────────────────────────
    all_results: list[dict] = []

    for idx, (seq_name, splat_3dgs_cpu) in enumerate(splat_cache, 1):
        print(f"  [{idx:3d}/{n_seqs}] {seq_name} …", end=" ", flush=True)
        res = evaluate_sequence(
            seq_name=seq_name,
            splat_3dgs_cpu=splat_3dgs_cpu,
            model=model,
            out_type=out_type,
            max_points=max_points,
            n_views=args.n_views,
            rng=rng,
            cache_dir=cache_dir,
            loss_cfg=cfg.train.loss,
            subset_lists_filename=str(cfg.data.subset_lists_filename),
            dataset_root=str(cfg.data.dataset_root),
            image_height=int(cfg.data.image_height),
            image_width=int(cfg.data.image_width),
            split=args.split,
            device=device,
            vis_dir=vis_dir,
        )
        all_results.append(res)
        step_psns = "  ".join(
            f"2DGS@{s}={res[f'psnr_2dgs_{s}']:.2f}" for s in STEP_COUNTS)
        print(f"3DGS={res['psnr_3dgs']:.2f}  {step_psns}  Model={res['psnr_model']:.2f}")

    # ── aggregate ──────────────────────────────────────────────────────────────
    def _mean(key):
        vals = [r[key] for r in all_results if r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    print()
    W = 84
    print("─" * W)
    print(f"  Results  ({len(all_results)} seqs, {args.n_views} views, split={args.split})")
    print("─" * W)
    print(f"  {'Method':<24}  {'PSNR (dB) ↑':>11}  {'Normal cos ↑':>12}  "
          f"{'Render ms':>10}  {'Fit/Infer ms':>12}")
    print("  " + "─" * (W - 2))

    def _row(name, psnr_k, normal_k, render_k, fit_k=None):
        psnr   = _mean(psnr_k)
        normal = _mean(normal_k)
        rms    = _mean(render_k)
        fms    = _mean(fit_k) if fit_k else None
        fms_s  = f"{fms:>12.0f}" if fms is not None else f"{'—':>12}"
        print(f"  {name:<24}  {psnr:>11.4f}  {normal:>12.4f}  {rms:>10.2f}  {fms_s}")

    _row("3DGS (dataset)",
         "psnr_3dgs",   "normal_3dgs",   "render_ms_3dgs")
    for steps in STEP_COUNTS:
        _row(f"2DGS @ {steps:4d} steps",
             f"psnr_2dgs_{steps}", f"normal_2dgs_{steps}",
             f"render_ms_2dgs_{steps}", f"fit_ms_2dgs_{steps}")
    _row(f"Model ({out_type.upper()})",
         "psnr_model",  "normal_model",  "render_ms_model", "model_ms")

    print("─" * W)
    print()
    print("  Normal cos: cosine similarity in camera frame vs depth-derived GT normals")
    print("              (higher = better, 1.0 = perfect).")
    print("  Fit/Infer ms: cumulative 2DGS optimisation time or model forward-pass time")
    print(f"                per sequence; cached to {cache_dir}/.")

    # ── optional CSV ──────────────────────────────────────────────────────────
    if args.out_csv:
        out_path = Path(args.out_csv)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Per-sequence CSV → {out_path}")


if __name__ == "__main__":
    main()
