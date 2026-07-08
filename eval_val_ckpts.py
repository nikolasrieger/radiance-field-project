#!/usr/bin/env python3
"""Compute the full set of validation metrics for a range of checkpoints.

Rebuilds exactly the validation set used for the live wandb metrics during
training (train.build_val_set: "val" split, data.max_splats filtering,
num_val_sequences / num_val_views selection, seeded max_points subsampling)
and runs train.py's _validate for each requested checkpoint epoch, so the
numbers are directly comparable to the val/* curves in wandb.

All metrics are always computed: the training-objective losses (loss, photo,
normal, depth — with the run's lambdas), PSNR, SSIM, LPIPS, depth L1/L2
abs/rel error, and (for 2dgs runs) the normal angular error — regardless of
which supervision terms were enabled during training.

Usage
-----
    python eval_val_ckpts.py runs/my_run                       # all ckpt_*.pt
    python eval_val_ckpts.py runs/my_run --epochs 10:100:10    # inclusive range
    python eval_val_ckpts.py runs/my_run --epochs 5,20,10:50:10 --final

Results are printed and written to <run_dir>/val_metrics.csv (see --out).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from radiance_field_project.config import load_config
from radiance_field_project.data.datamodule import UCO3DDataModule
from radiance_field_project.models.model_factory import build_model
from radiance_field_project.training.train import _validate, build_val_set


class _Passthrough(nn.Module):
    """Identity 'model': render the input splats with no refinement.

    Used to emit raw-splat baseline rows (2dgs / 3dgs) through the exact same
    val set + metric code as the trained checkpoints, so they are directly
    comparable, column for column.
    """

    def reconstruct_splats(self, splats):
        return splats


def _parse_epochs(spec: str) -> list[int]:
    """Parse '5,20,100' / '10:100:10' (inclusive) / any comma-mix of both."""
    epochs: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            nums = [int(x) for x in part.split(":")]
            start, stop = nums[0], nums[1]
            step = nums[2] if len(nums) > 2 else 1
            epochs.update(range(start, stop + 1, step))
        else:
            epochs.add(int(part))
    return sorted(epochs)


def _discover_ckpts(run_dir: Path) -> dict[int, Path]:
    """Map epoch → checkpoint path for every numbered checkpoint in the run."""
    ckpts: dict[int, Path] = {}
    for p in run_dir.glob("ckpt_[0-9]*.pt"):
        m = re.search(r"ckpt_(\d+)\.pt$", p.name)
        if m:
            ckpts[int(m.group(1))] = p
    return ckpts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path,
                    help="Run directory containing config.yaml and ckpt_*.pt")
    ap.add_argument("--epochs", type=str, default=None,
                    help="Checkpoint epochs to evaluate: '5,20,100' and/or "
                         "inclusive ranges '10:100:10'. Default: all found.")
    ap.add_argument("--final", action="store_true",
                    help="Also evaluate ckpt_final.pt")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV path (default: <run_dir>/val_metrics.csv)")
    ap.add_argument("--device", type=str, default=None,
                    help="Override device (default: config value / cuda)")
    ap.add_argument("--num-val-sequences", type=int, default=None,
                    help="Override data.num_val_sequences (e.g. 10 to eval a "
                         "handful of val scenes). Default: config value.")
    ap.add_argument("--baseline", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Also append raw-splat baseline rows (--baseline / "
                         "--no-baseline). Default: on.")
    ap.add_argument("--baseline-types", type=str, default="3dgs,2dgs",
                    help="Representations to baseline (default: 3dgs,2dgs).")
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg = load_config(run_dir / "config.yaml")
    if args.num_val_sequences is not None:
        cfg.data.num_val_sequences = args.num_val_sequences
    device = torch.device(args.device or
                          (cfg.run.device if torch.cuda.is_available() else "cpu"))

    available = _discover_ckpts(run_dir)
    if args.epochs is None:
        selected = [(e, available[e]) for e in sorted(available)]
    else:
        selected = []
        for e in _parse_epochs(args.epochs):
            if e in available:
                selected.append((e, available[e]))
            else:
                print(f"[WARN] no checkpoint for epoch {e} "
                      f"(expected {run_dir / f'ckpt_{e:05d}.pt'}) — skipping")
    if args.final:
        final = run_dir / "ckpt_final.pt"
        if final.exists():
            selected.append((None, final))
        else:
            print(f"[WARN] {final} not found — skipping")
    if not selected:
        raise SystemExit("No checkpoints to evaluate.")

    output_type = cfg.model.output_type.value
    max_points  = cfg.data.max_points
    # Compute every metric regardless of the training lambdas: depth metrics
    # only need GT depth (always loadable); normal metrics need the 2dgs
    # rasterizer. Lambdas of 0 keep the aggregate "loss" identical to the
    # training-time validation objective.
    use_normals = output_type == "2dgs"
    use_depth   = True

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
    print(f"Building validation set "
          f"({cfg.data.num_val_sequences or 'all'} val sequences)...")
    val_cache, val_loader, n_frames, n_seqs = build_val_set(
        cfg, dm, output_type, run_dir / "gaussian_cache_val")
    if val_loader is None:
        raise SystemExit("No usable sequences in the 'val' split.")
    print(f"Validation set: {n_frames} frames across {n_seqs} sequences | "
          f"{len(selected)} checkpoints | device={device}\n")

    model = build_model(cfg.model).to(device)
    model.eval()

    # CSV is written incrementally (flushed after every checkpoint) so an
    # interrupted sweep keeps all completed rows. The header is fixed up front
    # to the 2dgs superset (base metrics + normal_err) so baseline rows for a
    # representation with fewer keys (3dgs has no normal_err) fit the same
    # columns — DictWriter leaves the missing cell blank.
    fieldnames = ["epoch", "ckpt", "loss", "photo", "normal", "depth",
                  "psnr", "ssim", "lpips", "normal_err",
                  "depth_l1_abs", "depth_l2_abs", "depth_l1_rel", "depth_l2_rel"]
    out_csv = args.out or (run_dir / "val_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        print("  " + "  ".join(f"{k:>12}" for k in fieldnames if k != "ckpt"))

        def emit(label: str, ckpt_name: str, result: dict) -> None:
            print("  " + f"{label:>12}  " + "  ".join(
                f"{result[k]:>12.6f}" if k in result else f"{'':>12}"
                for k in fieldnames if k not in ("epoch", "ckpt")))
            writer.writerow({"epoch": label, "ckpt": ckpt_name,
                             **{k: f"{v:.6f}" for k, v in result.items()}})
            f.flush()

        for epoch, ckpt in selected:
            state = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state)
            result = _validate(model, val_cache, val_loader, cfg.train.loss,
                               output_type, use_normals, use_depth, max_points,
                               device, seed=cfg.run.seed)
            emit("final" if epoch is None else str(epoch), ckpt.name, result)

        if args.baseline:
            passthrough = _Passthrough().to(device)
            for bt in (t.strip() for t in args.baseline_types.split(",") if t.strip()):
                # Reuse the already-built cache/loader when the baseline
                # representation matches the run's; otherwise build that
                # representation's val set in its own cache dir.
                if bt == output_type:
                    b_cache, b_loader = val_cache, val_loader
                else:
                    b_cache, b_loader, _, _ = build_val_set(
                        cfg, dm, bt, run_dir / f"gaussian_cache_val_{bt}")
                    if b_loader is None:
                        print(f"[WARN] no usable 'val' sequences for {bt} "
                              f"baseline — skipping")
                        continue
                result = _validate(passthrough, b_cache, b_loader, cfg.train.loss,
                                   bt, bt == "2dgs", use_depth, max_points,
                                   device, seed=cfg.run.seed)
                emit(f"baseline_{bt}", "-", result)
    print(f"\nResults saved → {out_csv}")


if __name__ == "__main__":
    main()
