#!/usr/bin/env python3
"""Evaluate every checkpoint in each *_30 run across N_SCENES × N_VIEWS.

For each run:
  - finds all ckpt_*.pt + ckpt_final.pt sorted by epoch
  - picks N_SCENES sequences (one per seed 0..N_SCENES-1)
  - caches splat tensors and N_VIEWS frames per sequence once
  - swaps model weights per checkpoint (no dataset reload)
  - prints a per-run epoch-progression table and a final cross-run summary
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from radiance_field_project.config import load_config
from radiance_field_project.config.schema import ModelType, NormalLossType, NormalRegion
from radiance_field_project.data.collate import uco3d_collate
from radiance_field_project.data.datamodule import UCO3DDataModule, SequenceBatchSampler
from radiance_field_project.data.dataset import make_uco3d_dataset
from radiance_field_project.models.model_factory import build_model
from radiance_field_project.utils.rendering import (
    render_view, world_normals_to_camera, get_camera_params,
)
from radiance_field_project.training.train import (
    _cache_splat, _to_device, _subsample,
    _depth_loss, _photometric_loss,
)
from evaluate_splat import _build_model_from_state, _slice_camera, _frame_dict_with_depth

# ── config ────────────────────────────────────────────────────────────────────
RUNS_DIR = Path(__file__).resolve().parent / "runs"
N_SCENES = 5
N_VIEWS  = 5   # views per scene (n_eval per call to evaluate_splat logic)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ckpt_epoch(path: Path) -> int:
    """Return sort key: numbered epoch or inf for ckpt_final."""
    m = re.search(r"ckpt_(\d+)", path.name)
    return int(m.group(1)) if m else 10**9


def _ckpt_label(path: Path) -> str:
    m = re.search(r"ckpt_(\d+)", path.name)
    return m.group(1) if m else "final"


def _load_scene(cfg, seq_name: str, device, out_type: str, max_points: int,
                use_depth: bool, use_normal: bool):
    """Load splat + up to N_VIEWS frames for one sequence. Returns dict."""
    max_splats = int(cfg.data.max_splats) if cfg.data.max_splats else None

    # ── splat ─────────────────────────────────────────────────────────────────
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
    loader = splat_dm.splat_only_loader(max_splats=max_splats)
    seq_map = {
        loader.dataset[b[0]].sequence_name: b[0]
        for b in loader.batch_sampler._batches
    }
    raw = loader.dataset[seq_map[seq_name]].sequence_gaussian_splats
    splat_in  = _to_device(_subsample(_cache_splat(raw, output_type=out_type),  max_points), device)

    # ── frames ────────────────────────────────────────────────────────────────
    img_ds = make_uco3d_dataset(
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
    img_loader = DataLoader(
        img_ds,
        batch_sampler=SequenceBatchSampler(img_ds, shuffle=False),
        num_workers=0,
        collate_fn=uco3d_collate,
    )
    frames = []
    for batch in img_loader:
        T = batch["image_rgb"].shape[0]
        for i in range(min(N_VIEWS, T)):
            frames.append(_frame_dict_with_depth(batch, i))
        break

    return {"splat_in": splat_in, "frames": frames, "seq_name": seq_name}


def _eval_scene(model, scene: dict, out_type: str, loss_cfg, device,
                use_depth: bool, use_normal: bool) -> dict:
    """Run model on one scene, return averaged losses."""
    with torch.no_grad():
        splat_pred = model.reconstruct_splats(scene["splat_in"])

    total_photo = total_depth = total_normal = 0.0
    n = 0
    for frame in scene["frames"]:
        has_depth = "depth_map" in frame

        if use_normal and has_depth:
            vmats, K, gt_image, fg_mask, h, w, _, gt_normals = \
                get_camera_params(frame, device, ret_normals=True)
            rendered, normals_world = render_view(
                splat_pred, vmats, K, w, h,
                output_type=out_type, device=device, ret_normals=True,
            )
            normals_cam = world_normals_to_camera(normals_world, vmats)
            depth_valid = (gt_normals.norm(dim=1, keepdim=True) > 0.5).float()
            if loss_cfg.normal_region == NormalRegion.foreground:
                normal_mask = (fg_mask > 0.5).float() * depth_valid
            else:
                normal_mask = depth_valid
            if loss_cfg.normal_loss_type == NormalLossType.cosine:
                dot = (normals_cam * gt_normals).sum(dim=1, keepdim=True).clamp(-1, 1)
                normal_loss = ((1.0 - dot) * normal_mask).mean().item()
            else:
                normal_loss = F.l1_loss(
                    normals_cam * normal_mask, gt_normals * normal_mask).item()
            total_normal += normal_loss
        else:
            vmats, K, gt_image, fg_mask, h, w = get_camera_params(frame, device)
            rendered, rendered_depth = render_view(
                splat_pred, vmats, K, w, h,
                output_type=out_type, device=device, ret_depth=True,
            )

        total_photo += _photometric_loss(rendered, gt_image, fg_mask, loss_cfg).item()

        if has_depth and (use_depth or not use_normal):
            if use_normal and has_depth:
                _, rendered_depth = render_view(
                    splat_pred, vmats, K, w, h,
                    output_type=out_type, device=device, ret_depth=True,
                )
            dsa = frame["depth_scale_adjustment"]
            if hasattr(dsa, "to"):
                dsa = dsa.to(device).view(-1, 1, 1, 1).float()
            gt_depth   = frame["depth_map"].to(device) * dsa
            depth_mask = (frame["depth_mask"].to(device) > 0.5).float()
            total_depth += _depth_loss(rendered_depth, gt_depth, depth_mask).item()

        n += 1

    avg_photo  = total_photo  / max(n, 1)
    avg_depth  = total_depth  / max(n, 1)
    avg_normal = total_normal / max(n, 1)
    full = (avg_photo
            + loss_cfg.lambda_depth  * avg_depth
            + loss_cfg.lambda_normal * avg_normal)
    return {"photo": avg_photo, "depth": avg_depth, "normal": avg_normal, "full": full}


def fmt(v: float | None, w: int = 10) -> str:
    return f"{v:.6f}".center(w) if v is not None else "—".center(w)


# ── main ──────────────────────────────────────────────────────────────────────

runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.endswith("_30"))
if not runs:
    sys.exit(f"No *_30 runs found in {RUNS_DIR}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_rows: list[dict] = []   # best (final) checkpoint per run

for run in runs:
    ckpts = sorted(
        list(run.glob("ckpt_[0-9]*.pt")) + list(run.glob("ckpt_final.pt")),
        key=_ckpt_epoch,
    )
    if not ckpts:
        print(f"\n[SKIP] {run.name} — no checkpoints")
        continue

    cfg      = load_config(run / "config.yaml")
    loss_cfg = cfg.train.loss
    out_type = cfg.model.output_type.value
    is_2dgs  = out_type == "2dgs"
    max_pts  = int(cfg.data.max_points) if cfg.data.max_points else 0
    use_depth  = True                # always evaluate all losses
    use_normal = is_2dgs

    # ── pick N_SCENES sequences ───────────────────────────────────────────────
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
    all_batches = list(splat_dm.splat_only_loader(max_splats=max_splats).batch_sampler._batches)
    seq_names: list[str] = []
    seen: set[str] = set()
    for seed in range(N_SCENES):
        b = random.Random(seed).choice(all_batches)
        name = splat_dm.splat_only_loader(max_splats=max_splats).dataset[b[0]].sequence_name
        if name not in seen:
            seq_names.append(name)
            seen.add(name)
    seq_names = seq_names[:N_SCENES]

    print(f"\n{'═'*60}")
    print(f"  {run.name}")
    print(f"  Sequences: {', '.join(seq_names)}")
    print(f"  Checkpoints: {len(ckpts)}   ({N_SCENES} scenes × {N_VIEWS} views each)")
    print(f"{'═'*60}")

    # ── pre-load scenes ───────────────────────────────────────────────────────
    print("  Loading scenes...", end=" ", flush=True)
    scenes = []
    for sn in seq_names:
        try:
            scenes.append(_load_scene(cfg, sn, device, out_type, max_pts,
                                      use_depth, use_normal))
        except Exception as e:
            print(f"\n  [WARN] skipping {sn}: {e}")
    print(f"loaded {len(scenes)}.")

    if not scenes:
        print("  No scenes loaded, skipping run.")
        continue

    # ── build model skeleton from first checkpoint ────────────────────────────
    first_state = torch.load(ckpts[0], map_location=device, weights_only=True)
    model = _build_model_from_state(cfg.model, first_state).to(device)
    model.eval()

    # ── table header ─────────────────────────────────────────────────────────
    cw = 10
    cols = ["Photo", "Depth", "Normal", "Full"] if is_2dgs else ["Photo", "Depth", "Full"]
    header = f"  {'Epoch':>8}  " + "  ".join(f"{c:^{cw}}" for c in cols)
    sep    = "  " + "─" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)

    run_rows: list[dict] = []

    for ckpt in ckpts:
        label = _ckpt_label(ckpt)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)

        # average across scenes
        agg = {"photo": 0.0, "depth": 0.0, "normal": 0.0, "full": 0.0}
        for scene in scenes:
            try:
                r = _eval_scene(model, scene, out_type, loss_cfg, device,
                                use_depth, use_normal)
                for k in agg:
                    agg[k] += r[k]
            except Exception as e:
                print(f"\n  [WARN] {ckpt.name} on {scene['seq_name']}: {e}")
        n_ok = len(scenes)
        row = {k: agg[k] / n_ok for k in agg}
        row["label"] = label
        run_rows.append(row)

        vals = [fmt(row["photo"], cw), fmt(row["depth"], cw)]
        if is_2dgs:
            vals.append(fmt(row["normal"], cw))
        vals.append(fmt(row["full"], cw))
        print(f"  {label:>8}  " + "  ".join(vals))

    print(sep)

    if run_rows:
        final = run_rows[-1]
        summary_rows.append({"run": run.name, **{k: final[k] for k in ("photo","depth","normal","full")}, "is_2dgs": is_2dgs})

# ── final cross-run summary ───────────────────────────────────────────────────
if not summary_rows:
    sys.exit()

col_run = max(len(r["run"]) for r in summary_rows)
cw = 10
header = "  ".join([
    f"{'Run':<{col_run}}",
    f"{'Photo':^{cw}}", f"{'Depth':^{cw}}", f"{'Normal':^{cw}}", f"{'Full':^{cw}}",
])
sep = "─" * len(header)

print(f"\n\n{sep}")
print(f"  FINAL CHECKPOINT SUMMARY  ({N_SCENES} scenes × {N_VIEWS} views)")
print(sep)
print(header)
print(sep)
for r in summary_rows:
    normal_str = fmt(r.get("normal"), cw) if r.get("is_2dgs") else "—".center(cw)
    print("  ".join([
        f"{r['run']:<{col_run}}",
        fmt(r.get("photo"), cw),
        fmt(r.get("depth"), cw),
        normal_str,
        fmt(r.get("full"),  cw),
    ]))
print(sep)
