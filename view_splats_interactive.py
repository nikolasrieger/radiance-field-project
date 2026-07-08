#!/usr/bin/env python3
"""Interactive side-by-side 3DGS vs 2DGS viewer.

Picks a random sequence, runs the trained model, then opens a viser web viewer
showing the input 3DGS and predicted splat side by side. Drag / orbit the
camera to compare both from the same angle. GUI controls let you toggle
visibility and adjust the gap.

Usage
-----
    cd radiance-field-project
    uv run python view_splats_interactive.py --run runs/minipointnet_2dgs_...
    uv run python view_splats_interactive.py --run runs/... --seed 7 --port 8080
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import viser

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from radiance_field_project.config import load_config
from radiance_field_project.data.datamodule import UCO3DDataModule
from radiance_field_project.models.model_factory import build_model
from radiance_field_project.training.train import _cache_splat, _to_device, _subsample
from uco3d import GaussianSplats

_SH_C0 = 0.28209479177387814


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


def _build_covariances(splats: GaussianSplats) -> np.ndarray:
    """(N,3) log-scales + (N,4) quats [w,x,y,z] → (N,3,3) covariances."""
    s = torch.exp(splats.scales)                          # (N, 3)
    q = F.normalize(splats.quats, p=2, dim=-1)            # (N, 4)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)                          # (N, 3, 3)
    S2 = torch.diag_embed(s * s)                          # (N, 3, 3)
    cov = R @ S2 @ R.transpose(-1, -2)                    # (N, 3, 3)
    return cov.detach().cpu().float().numpy()


def _to_viser_arrays(splats: GaussianSplats):
    """Return (centers, covariances, rgbs, opacities) as contiguous float32 arrays."""
    centers     = splats.means.detach().cpu().float().numpy()
    covariances = _build_covariances(splats)
    rgbs        = (_SH_C0 * splats.sh0 + 0.5).clamp(0, 1).detach().cpu().float().numpy()
    opacities   = torch.sigmoid(splats.opacities.flatten()).detach().cpu().float().numpy()[:, None]
    # viser requires C-contiguous arrays
    return (np.ascontiguousarray(centers),
            np.ascontiguousarray(covariances),
            np.ascontiguousarray(rgbs),
            np.ascontiguousarray(opacities))


def _scene_extent(means: torch.Tensor) -> float:
    """Largest axis-aligned extent of a point cloud."""
    lo = means.min(dim=0).values
    hi = means.max(dim=0).values
    return float((hi - lo).max())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",  required=True,
                   help="Training run directory (must contain config.yaml)")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint filename inside --run")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for sequence selection (default: 0)")
    p.add_argument("--port", type=int, default=8080,
                   help="Viser web server port (default: 8080)")
    p.add_argument("--gap",  type=float, default=1.3,
                   help="Initial gap multiplier: gap = scene_extent × this (default: 1.3)")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Dataset split to pick the sequence from (default: train)")
    return p.parse_args()


def main():
    args    = parse_args()
    run_dir = Path(args.run)
    ckpt    = _find_ckpt(run_dir, args.ckpt)
    cfg     = load_config(run_dir / "config.yaml")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Run:    {run_dir.name}")
    print(f"Ckpt:   {ckpt.name}")
    print(f"Device: {device}")

    # ── model ──────────────────────────────────────────────────────────────────
    state = torch.load(ckpt, map_location=device, weights_only=True)
    # Detect SA1 input width from checkpoint to handle old xyz-only checkpoints.
    ckpt_in_ch = None
    if "sa1.mlp_convs.0.weight" in state:
        ckpt_in_ch = state["sa1.mlp_convs.0.weight"].shape[1]
    model = build_model(cfg.model, encoder_in_channel=ckpt_in_ch).to(device)
    model.load_state_dict(state)
    model.eval()

    out_type   = cfg.model.output_type.value   # "2dgs" | "3dgs"
    max_points = int(cfg.data.max_points) if cfg.data.max_points else 0

    # ── pick a random sequence ─────────────────────────────────────────────────
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
    splat_loader = splat_dm.splat_only_loader(max_splats=max_splats,
                                              subsets=(args.split,))
    all_batches  = list(splat_loader.batch_sampler._batches)

    rng          = random.Random(args.seed)
    chosen_batch = rng.choice(all_batches)
    frame_data   = splat_loader.dataset[chosen_batch[0]]
    seq_name     = frame_data.sequence_name
    raw_splat    = frame_data.sequence_gaussian_splats
    print(f"Sequence: {seq_name}  ({raw_splat.means.shape[0]:,} raw points)")

    # ── prepare and run model ─────────────────────────────────────────────────
    splat_3dgs_cpu = _cache_splat(raw_splat, output_type='3dgs')
    splat_in_cpu   = _cache_splat(raw_splat, output_type=out_type)

    splat_3dgs = _to_device(_subsample(splat_3dgs_cpu, max_points), device)
    splat_in   = _to_device(_subsample(splat_in_cpu,   max_points), device)

    with torch.no_grad():
        splat_pred = model.reconstruct_splats(splat_in)

    # ── compute initial gap ────────────────────────────────────────────────────
    means_3dgs = splat_3dgs.means.detach().cpu()
    extent     = _scene_extent(means_3dgs)
    gap_init   = extent * args.gap

    print(f"3DGS:      {means_3dgs.shape[0]:,} Gaussians")
    print(f"{out_type.upper()} pred: {splat_pred.means.shape[0]:,} Gaussians")
    print(f"Scene extent: {extent:.3f}  →  initial gap: {gap_init:.3f}")

    # ── convert to viser arrays ────────────────────────────────────────────────
    c3, cov3, rgb3, op3 = _to_viser_arrays(splat_3dgs)
    cp, covp, rgbp, opp = _to_viser_arrays(splat_pred)

    # ── launch viser ──────────────────────────────────────────────────────────
    server = viser.ViserServer(port=args.port)
    print(f"\n  Open in browser → http://localhost:{args.port}")
    print("  Left-drag: orbit  |  right-drag: pan  |  scroll: zoom")
    print("  Ctrl+C to quit.\n")

    # Place 3DGS at its original world position; shift predicted cloud to the right.
    h3 = server.scene.add_gaussian_splats(
        "/splats/3dgs",
        centers=c3,
        covariances=cov3,
        rgbs=rgb3,
        opacities=op3,
    )
    hp = server.scene.add_gaussian_splats(
        f"/splats/{out_type}_pred",
        centers=cp,
        covariances=covp,
        rgbs=rgbp,
        opacities=opp,
        position=(gap_init, 0.0, 0.0),
    )

    # Text labels floating below each cloud
    label_y = float(means_3dgs[:, 1].min()) - 0.15 * extent
    cx_3dgs = float(means_3dgs[:, 0].mean())
    cx_pred = float(splat_pred.means.detach().cpu()[:, 0].mean()) + gap_init
    server.scene.add_label("/labels/3dgs", text="3DGS  (input)",
                           position=(cx_3dgs, label_y, 0.0))
    server.scene.add_label(f"/labels/{out_type}_pred",
                           text=f"{out_type.upper()}  (predicted)",
                           position=(cx_pred, label_y, 0.0))

    # ── GUI ────────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Display"):
        cb_3dgs = server.gui.add_checkbox("Show 3DGS",                   initial_value=True)
        cb_pred = server.gui.add_checkbox(f"Show {out_type.upper()} pred", initial_value=True)
        sl_gap  = server.gui.add_slider(
            "Gap (× extent)", min=0.0, max=5.0, step=0.05, initial_value=args.gap,
        )

    with server.gui.add_folder("Info", expand_by_default=False):
        server.gui.add_markdown(
            f"**Sequence:** `{seq_name}`\n\n"
            f"**3DGS:** {c3.shape[0]:,} Gaussians\n\n"
            f"**{out_type.upper()} pred:** {cp.shape[0]:,} Gaussians\n\n"
            f"**Run:** `{run_dir.name}`\n\n"
            f"**Ckpt:** `{ckpt.name}`"
        )

    @cb_3dgs.on_update
    def _(_): h3.visible = cb_3dgs.value

    @cb_pred.on_update
    def _(_): hp.visible = cb_pred.value

    @sl_gap.on_update
    def _(_):
        new_gap = extent * sl_gap.value
        hp.position = (new_gap, 0.0, 0.0)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
