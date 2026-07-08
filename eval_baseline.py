#!/usr/bin/env python3
"""Compute val metrics for the *raw* input splats (no model refinement).

Reference / lower-bound numbers to sit alongside the trained-checkpoint table
from ``eval_val_ckpts.py``. Uses the identical validation set (same
``build_val_set``) and the identical metric code (``_validate``), but swaps the
trained network for an identity passthrough whose ``reconstruct_splats`` returns
the cached splats unchanged. So the rows are directly comparable, column for
column, to ``val_metrics*.csv``.

Two representations are evaluated by default:

  - ``3dgs``  the original UCO3D Gaussians rendered directly (the model's input).
  - ``2dgs``  those Gaussians canonicalized to 2D (what the network starts from
              before it refines them). Adds the ``normal_err`` column, since the
              2DGS rasterizer produces normals.

Each representation builds its own gaussian cache with the matching
``output_type`` (separate ``gaussian_cache_val_<type>`` dirs, so the trained-run
2dgs cache is never clobbered).

Usage
-----
    python eval_baseline.py runs/pointnet2_l2_cos --num-val-sequences 10
    python eval_baseline.py runs/pointnet2_l2_cos --types 2dgs        # one only
    python eval_baseline.py runs/my_run --out runs/my_run/baseline.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from radiance_field_project.config import load_config
from radiance_field_project.data.datamodule import UCO3DDataModule
from radiance_field_project.training.train import _validate, build_val_set


class _Passthrough(nn.Module):
    """Identity 'model': render the input splats with no refinement."""

    def reconstruct_splats(self, splats):
        return splats


def _eval_type(cfg, dm, output_type, run_dir, max_points, device):
    """Build the val set for ``output_type`` and score the raw splats on it."""
    use_normals = output_type == "2dgs"
    use_depth   = True

    print(f"[{output_type}] building validation set "
          f"({cfg.data.num_val_sequences or 'all'} val sequences)...")
    val_cache, val_loader, n_frames, n_seqs = build_val_set(
        cfg, dm, output_type, run_dir / f"gaussian_cache_val_{output_type}")
    if val_loader is None:
        print(f"[{output_type}] no usable sequences in 'val' split — skipping")
        return None
    print(f"[{output_type}] {n_frames} frames across {n_seqs} sequences")

    model = _Passthrough().to(device)
    return _validate(model, val_cache, val_loader, cfg.train.loss,
                     output_type, use_normals, use_depth, max_points,
                     device, seed=cfg.run.seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path,
                    help="Run directory containing config.yaml (for val-set + "
                         "loss/lambda + seed settings).")
    ap.add_argument("--types", type=str, default="3dgs,2dgs",
                    help="Comma list of representations to score "
                         "(default: 3dgs,2dgs).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV (default: <run_dir>/val_metrics_baseline.csv)")
    ap.add_argument("--device", type=str, default=None,
                    help="Override device (default: config value / cuda)")
    ap.add_argument("--num-val-sequences", type=int, default=None,
                    help="Override data.num_val_sequences (e.g. 10). Must match "
                         "the eval_val_ckpts.py run to be comparable.")
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg = load_config(run_dir / "config.yaml")
    if args.num_val_sequences is not None:
        cfg.data.num_val_sequences = args.num_val_sequences
    device = torch.device(args.device or
                          (cfg.run.device if torch.cuda.is_available() else "cpu"))
    max_points = cfg.data.max_points
    types = [t.strip() for t in args.types.split(",") if t.strip()]

    dm = UCO3DDataModule(
        subset_lists_filename=cfg.data.subset_lists_filename,
        dataset_root=cfg.data.dataset_root,
        num_workers=cfg.data.num_workers,
        image_height=cfg.data.image_height,
        image_width=cfg.data.image_width,
        load_masks=cfg.data.load_masks,
        load_depths=True,
        load_gaussian_splats=True,
        pick_sequences=list(cfg.data.pick_sequences),
    )

    results = []
    for output_type in types:
        r = _eval_type(cfg, dm, output_type, run_dir, max_points, device)
        if r is not None:
            results.append((output_type, r))
    if not results:
        raise SystemExit("Nothing evaluated.")

    # Union of metric keys (2dgs has normal_err, 3dgs does not) so every row
    # fits one header; missing cells are left blank.
    keys: list[str] = []
    for _, r in results:
        for k in r:
            if k not in keys:
                keys.append(k)

    out_csv = args.out or (run_dir / "val_metrics_baseline.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["baseline", *keys])
        writer.writeheader()
        print("\n  " + "  ".join(f"{k:>12}" for k in ["baseline", *keys]))
        for output_type, r in results:
            print("  " + f"{output_type:>12}  " + "  ".join(
                f"{r[k]:>12.6f}" if k in r else f"{'':>12}" for k in keys))
            writer.writerow({"baseline": output_type,
                             **{k: f"{v:.6f}" for k, v in r.items()}})
    print(f"\nResults saved → {out_csv}")


if __name__ == "__main__":
    main()
