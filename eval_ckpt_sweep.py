#!/usr/bin/env python3
"""Sweep all saved checkpoints for the specified runs and report PSNR + Normal cos over time.

For each run:
  - finds all ckpt_*.pt + ckpt_final.pt sorted by epoch
  - evaluates the model on N_SEQS × N_VIEWS train-split frames
    (same sequence selection as eval_benchmark.py --train N_SEQS)
  - reports PSNR (dB) and Normal cos per checkpoint
  - writes a summary CSV

Scene data is loaded once per run; only model weights are swapped per checkpoint.
"""
from __future__ import annotations

import csv
import random
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from radiance_field_project.config import load_config
from radiance_field_project.data.collate import uco3d_collate
from radiance_field_project.data.datamodule import UCO3DDataModule, SequenceBatchSampler
from radiance_field_project.data.dataset import make_uco3d_dataset
from radiance_field_project.utils.rendering import (
    get_camera_params, render_view, world_normals_to_camera, depth_to_normals,
)
from radiance_field_project.training.train import (
    _cache_splat, _to_device, _subsample,
)
from eval_benchmark import (
    _build_model_from_state, _psnr, _normal_cosine, _frame_dict,
)

# ── settings (match existing benchmark logs) ──────────────────────────────────
RUNS = [
    "minipointnet_2dgs_n_full_0.0_maxp_30",
    "minipointnet_2dgs_n_full_0.1_maxp_30",
    "minipointnet_2dgs_n_full_0.01_maxp_30",
    "minipointnet_2dgs_n_full_0.05_l1_maxp_30",
    "minipointnet_2dgs_n_full_0.05_maxp_30",
]
N_SEQS  = 4    # first N sequences from the train splat loader (matches --train 4)
N_VIEWS = 10   # random views per sequence
SEED    = 42   # RNG seed for view sampling
SPLIT   = "train"
OUT_CSV = Path(__file__).resolve().parent / "ckpt_sweep_results.csv"

RUNS_DIR = Path(__file__).resolve().parent / "runs"


# ── helpers ───────────────────────────────────────────────────────────────────

def _ckpt_epoch(path: Path) -> int:
    m = re.search(r"ckpt_(\d+)", path.name)
    return int(m.group(1)) if m else 10**9


def _ckpt_label(path: Path) -> str:
    m = re.search(r"ckpt_(\d+)", path.name)
    return m.group(1) if m else "final"


def _load_view_data(cfg, seq_name: str, n_views: int, seed: int,
                    device) -> list[tuple]:
    """Load GT images + normals for one sequence. Returns list of view tuples."""
    img_ds = make_uco3d_dataset(
        subsets=(SPLIT,),
        subset_lists_filename=str(cfg.data.subset_lists_filename),
        dataset_root=str(cfg.data.dataset_root),
        load_images=True,
        load_depths=True,
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
    batch = next(iter(img_loader))
    T = batch["image_rgb"].shape[0]
    rng = random.Random(seed)
    all_idx = list(range(T))
    chosen = all_idx if len(all_idx) <= n_views else sorted(rng.sample(all_idx, n_views))

    view_data: list[tuple] = []
    for i in chosen:
        frame = _frame_dict(batch, i)
        vmats, K, gt_image, fg_mask, h, w, _, gt_normals = \
            get_camera_params(frame, device, ret_normals=True)
        view_data.append((vmats, K, gt_image, fg_mask, gt_normals, h, w))
    return view_data


def _eval_model_on_scenes(model, scenes: list[dict], out_type: str,
                          device) -> tuple[float, float]:
    """Return (mean_psnr, mean_normal_cos) across all scenes and views."""
    psnr_total = 0.0
    normal_sum = 0.0
    normal_n   = 0
    view_count = 0

    with torch.no_grad():
        for scene in scenes:
            splat_pred = model.reconstruct_splats(scene["splat_in"])

            for vmats, K, gt_image, fg_mask, gt_normals, h, w in scene["view_data"]:
                if out_type == "2dgs":
                    rgb, normals_world = render_view(
                        splat_pred, vmats, K, w, h,
                        output_type="2dgs", device=device, ret_normals=True,
                    )
                    normals_cam = world_normals_to_camera(normals_world, vmats)
                else:
                    rgb, depth = render_view(
                        splat_pred, vmats, K, w, h,
                        output_type=out_type, device=device, ret_depth=True,
                    )
                    normals_cam = depth_to_normals(depth, K, (depth > 0).float())

                psnr_total += _psnr(rgb, gt_image, fg_mask)
                cos = _normal_cosine(normals_cam, gt_normals, fg_mask)
                if cos == cos:  # not NaN
                    normal_sum += cos
                    normal_n   += 1
                view_count += 1

    psnr_mean   = psnr_total / max(view_count, 1)
    normal_mean = normal_sum / max(normal_n, 1)
    return psnr_mean, normal_mean


# ── main ──────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_csv_rows: list[dict] = []

for run_name in RUNS:
    run_dir = RUNS_DIR / run_name
    if not run_dir.exists():
        print(f"\n[SKIP] {run_name} — directory not found")
        continue

    ckpts = sorted(
        list(run_dir.glob("ckpt_[0-9]*.pt")) + list(run_dir.glob("ckpt_final.pt")),
        key=_ckpt_epoch,
    )
    if not ckpts:
        print(f"\n[SKIP] {run_name} — no checkpoints")
        continue

    cfg        = load_config(run_dir / "config.yaml")
    out_type   = cfg.model.output_type.value
    max_splats = int(cfg.data.max_splats) if cfg.data.max_splats else None
    max_points = int(cfg.data.max_points) if cfg.data.max_points else 0

    # Determine the first N_SEQS train sequences (same order as eval_benchmark.py)
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
    splat_loader = splat_dm.splat_only_loader(max_splats=max_splats, subsets=(SPLIT,))
    batches      = list(splat_loader.batch_sampler._batches)
    n_seqs       = min(N_SEQS, len(batches))
    seq_names    = [splat_loader.dataset[b[0]].sequence_name for b in batches[:n_seqs]]

    print(f"\n{'═'*72}")
    print(f"  {run_name}")
    print(f"  Sequences : {', '.join(seq_names)}")
    print(f"  Checkpoints: {len(ckpts)}  |  {n_seqs} seqs × {N_VIEWS} views  |  split={SPLIT}")
    print(f"{'═'*72}")

    # Pre-load input splats and view data once
    print("  Loading scene data ...", flush=True)
    scenes: list[dict] = []
    for b, sn in zip(batches[:n_seqs], seq_names):
        try:
            raw      = splat_loader.dataset[b[0]].sequence_gaussian_splats
            splat_in = _to_device(
                _subsample(_cache_splat(raw, output_type=out_type), max_points), device)
            view_data = _load_view_data(cfg, sn, N_VIEWS, SEED, device)
            scenes.append({"seq_name": sn, "splat_in": splat_in, "view_data": view_data})
            print(f"    loaded {sn} ({len(view_data)} views)", flush=True)
        except Exception as e:
            print(f"  [WARN] failed to load {sn}: {e}")

    if not scenes:
        print("  No scenes loaded — skipping run.")
        continue

    # Build model skeleton from first checkpoint
    first_state = torch.load(ckpts[0], map_location=device, weights_only=True)
    model = _build_model_from_state(cfg.model, first_state).to(device)
    model.eval()

    print(f"\n  {'Epoch':>8}  {'PSNR (dB)':>10}  {'Normal cos':>10}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}")

    for ckpt in ckpts:
        label = _ckpt_label(ckpt)
        try:
            epoch = int(label) if label != "final" else None
        except ValueError:
            epoch = None

        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)

        try:
            psnr_mean, normal_mean = _eval_model_on_scenes(model, scenes, out_type, device)
        except Exception as e:
            print(f"  {label:>8}  ERROR: {e}")
            continue

        print(f"  {label:>8}  {psnr_mean:>10.4f}  {normal_mean:>10.4f}", flush=True)

        all_csv_rows.append({
            "run":        run_name,
            "ckpt":       ckpt.name,
            "epoch":      epoch if epoch is not None else "",
            "psnr":       f"{psnr_mean:.6f}",
            "normal_cos": f"{normal_mean:.6f}",
        })

    print(f"  {'─'*8}  {'─'*10}  {'─'*10}")

# ── CSV output ────────────────────────────────────────────────────────────────
if all_csv_rows:
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "ckpt", "epoch", "psnr", "normal_cos"])
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\nResults saved → {OUT_CSV}")
else:
    print("\nNo results to save.")
