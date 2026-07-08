"""Config-driven training entry-point.

Usage
-----
    uv run training --config configs/pointnet2.yaml
    uv run training --config configs/minipointnet.yaml run.name=my_run train.lr=5e-4

Everything except --config is forwarded to OmegaConf as dotlist overrides.

Training is two-phase:
  1. Gaussian cache  — splat_only_loader converts all sequence splats once into
     disk-backed shards (reused across runs via the cache's index.json).
  2. Training loop   — image_only_loader (SequenceBatchSampler) drives each
     epoch; frame data (cameras / GT / normals / depth) is computed on the fly
     from the prefetched batches, nothing is cached to disk.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from uco3d import GaussianSplats
from uco3d.dataset_utils.data_types import Cameras

from radiance_field_project.config import parse_cli
from radiance_field_project.config.schema import LossType, NormalLossType, NormalRegion, Optimizer, OutputType, SchedulerType, WandbMode
from radiance_field_project.data.collate import uco3d_collate
from radiance_field_project.data.datamodule import UCO3DDataModule
from radiance_field_project.models.model_factory import build_model
from radiance_field_project.utils.gaussian_ops import to_2dgs_canonical
from radiance_field_project.utils.losses import reconstruction_loss, reconstruction_loss_l1
from radiance_field_project.utils import metrics
from radiance_field_project.utils.rendering import get_camera_params, render_view, world_normals_to_camera
from radiance_field_project.utils.splat_visualizer import SplatRenderVisualizer


# ── frame-slicing helpers ─────────────────────────────────────────────────────

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


def _slice_frame(batch: dict, i: int) -> dict:
    """Extract frame i from a sequence batch as a single-frame dict."""
    dsa = batch["depth_scale_adjustment"]
    frame = {
        "image_rgb":              batch["image_rgb"][i : i + 1],
        "fg_probability":         batch["fg_probability"][i : i + 1] if batch["fg_probability"] is not None else torch.ones_like(batch["image_rgb"][i : i + 1, :1]),
        "camera":                 _slice_camera(batch["camera"], i),
        "depth_scale_adjustment": dsa[i : i + 1] if torch.is_tensor(dsa) else dsa,
    }
    if "depth_map" in batch:
        frame["depth_map"]  = batch["depth_map"][i : i + 1]
        frame["depth_mask"] = batch["depth_mask"][i : i + 1]
    return frame


# ── gaussian cache helpers ────────────────────────────────────────────────────

_LINPRIM_DIST_TO_SCALE = 2.6


def _cache_splat(raw: GaussianSplats, output_type: str) -> GaussianSplats:
    """Convert a raw UCO3D GaussianSplats to a CPU float tensor.

    - ``'3dgs'``: scales stay as ``(N, 3)`` log-space.
    - ``'2dgs'``: canonicalize so the thin axis is at position 2.
    - ``'linprim'``: derive a single symmetric distance from the geometric
      mean of the 3DGS scales (scaled by DIST_TO_SCALE=2.6) and store as
      ``scales (N, 1)``.  ``render_view`` applies ``abs`` and expands to
      ``(N, 3)`` for the symmetric octahedron rasterizer.
    """
    means     = raw.means.float().cpu()
    sh0       = raw.sh0.float().cpu()
    scales    = raw.scales.float().cpu()
    quats     = raw.quats.float().cpu()
    opacities = raw.opacities.float().cpu()

    if output_type == "2dgs":
        scales2d, quats = to_2dgs_canonical(scales, quats)
        thin   = scales.min(dim=-1, keepdim=True).values
        scales = torch.cat([scales2d, thin], dim=-1)  # (N, 3) canonical
    elif output_type == "linprim":
        # Max 3DGS scale axis → linprim dist.  Using the max (rather than
        # geometric mean) avoids near-invisible octahedra for thin/small
        # Gaussians where the mean log-scale is very negative.
        dist   = torch.exp(scales.max(dim=-1, keepdim=True).values) * _LINPRIM_DIST_TO_SCALE
        scales = dist  # (N, 1)

    return GaussianSplats(means=means, sh0=sh0, scales=scales, quats=quats,
                          opacities=opacities, shN=None, fg_mask=None)


def _to_device(splats: GaussianSplats, device) -> GaussianSplats:
    return GaussianSplats(
        means=splats.means.to(device),
        sh0=splats.sh0.to(device),
        scales=splats.scales.to(device),
        quats=splats.quats.to(device),
        opacities=splats.opacities.to(device),
        shN=None, fg_mask=None,
    )


def _subsample(splats: GaussianSplats, max_points: int,
               generator: torch.Generator | None = None) -> GaussianSplats:
    if max_points <= 0 or splats.means.shape[0] <= max_points:
        return splats
    perm = torch.randperm(splats.means.shape[0], device=splats.means.device,
                          generator=generator)[:max_points]
    return GaussianSplats(
        means=splats.means[perm], sh0=splats.sh0[perm], scales=splats.scales[perm],
        quats=splats.quats[perm], opacities=splats.opacities[perm],
        shN=None, fg_mask=None,
    )


# ── training helpers ──────────────────────────────────────────────────────────

def _build_optimizer(params, cfg):
    if cfg.optimizer == Optimizer.adamw:
        return AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == Optimizer.adam:
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def _trainstate_path(ckpt_path: Path) -> Path:
    """Sidecar next to a model checkpoint holding optimizer/scheduler state.

    Kept as a separate file (rather than widening the ckpt_*.pt format) so the
    eval scripts that ``torch.load`` checkpoints as plain state dicts keep
    working, and so their ``ckpt_[0-9]*.pt`` globs don't pick it up.
    """
    return ckpt_path.with_name(ckpt_path.name.replace("ckpt_", "trainstate_", 1))


def _save_trainstate(ckpt_path: Path, optimizer, scheduler, epoch: int) -> None:
    torch.save(
        {"optimizer": optimizer.state_dict(),
         "scheduler": scheduler.state_dict() if scheduler is not None else None,
         "epoch": epoch},
        _trainstate_path(ckpt_path),
    )


def _resume_from_checkpoint(ckpt_path: Path, resume_epoch: int,
                            model, optimizer, scheduler, device) -> int:
    """Load a checkpoint for resuming; returns the epoch index to start at.

    ``resume_epoch`` >= 0 wins; otherwise the epoch is parsed from the
    filename (``ckpt_00042.pt`` was saved after epoch 42 → start at 43).
    Optimizer/scheduler state is restored from the ``trainstate_*.pt``
    sidecar when present; runs from before the sidecar existed resume with a
    fresh optimizer at the configured lr.
    """
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    ts_path = _trainstate_path(ckpt_path)
    if resume_epoch >= 0:
        start_epoch = resume_epoch
    else:
        m = re.search(r"ckpt_(\d+)", ckpt_path.name)
        if m is not None:
            start_epoch = int(m.group(1)) + 1
        elif ts_path.exists():
            start_epoch = torch.load(ts_path, map_location="cpu",
                                     weights_only=True)["epoch"] + 1
        else:
            raise ValueError(
                f"Cannot infer the resume epoch from '{ckpt_path.name}'; "
                "set train.resume_epoch explicitly.")

    if ts_path.exists():
        ts = torch.load(ts_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(ts["optimizer"])
        if scheduler is not None and ts.get("scheduler") is not None:
            scheduler.load_state_dict(ts["scheduler"])
        print(f"Resuming from {ckpt_path} at epoch {start_epoch} "
              f"(optimizer/scheduler state restored)")
    else:
        print(f"Resuming from {ckpt_path} at epoch {start_epoch} — no "
              f"{ts_path.name} found, optimizer/scheduler start fresh")
    return start_epoch


def _build_scheduler(optimizer, cfg):
    """Epoch-keyed scheduler. Uses absolute epoch counts (patience, not a
    fraction of train.epochs), so it stays correct even with epochs set to a
    very high placeholder value and training stopped manually."""
    if cfg.scheduler.type == SchedulerType.none:
        return None
    if cfg.scheduler.type == SchedulerType.plateau:
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience, threshold=cfg.scheduler.threshold,
            min_lr=cfg.scheduler.min_lr,
        )
    raise ValueError(f"Unknown scheduler type: {cfg.scheduler.type}")


def _depth_loss(rendered_depth: torch.Tensor, gt_depth: torch.Tensor,
                depth_mask: torch.Tensor) -> torch.Tensor:
    """Scale-normalised L1 depth loss masked to valid GT depth pixels."""
    n_valid   = depth_mask.sum().clamp(min=1.0)
    mean_depth = (gt_depth * depth_mask).sum() / n_valid
    abs_err   = (rendered_depth.nan_to_num(0.0) - gt_depth).abs() * depth_mask
    return abs_err.sum() / (n_valid * mean_depth.clamp(min=1e-6))


def _photometric_loss(rendered, gt_image, fg_mask, loss_cfg):
    mask = fg_mask if loss_cfg.recon_region == NormalRegion.foreground else None
    if loss_cfg.type == LossType.l2_masked:
        return reconstruction_loss(rendered, gt_image, mask=mask,
                                   lambda_ssim=loss_cfg.lambda_ssim)
    if loss_cfg.type == LossType.l1:
        return reconstruction_loss_l1(rendered, gt_image, mask=mask,
                                      lambda_ssim=loss_cfg.lambda_ssim)
    if loss_cfg.type == LossType.l1_masked:
        # Always mask the L1 term to the foreground, regardless of recon_region.
        return reconstruction_loss_l1(rendered, gt_image, mask=fg_mask,
                                      lambda_ssim=loss_cfg.lambda_ssim)
    if mask is not None:
        binary = (mask > 0.5).float()
        return ((rendered - gt_image).abs() * binary).sum() / binary.sum().clamp(min=1.0)
    return (rendered - gt_image).abs().mean()


def _frame_losses(recon, fd, loss_cfg, output_type, use_normals, use_depth, device,
                  return_extras=False):
    """Per-frame losses for ``recon`` against cached frame data ``fd``.

    Returns ``(frame_loss, photo_loss, normal_loss, depth_loss)`` as tensors;
    the normal/depth terms are zero tensors when their supervision is disabled.
    Shared by the training step and validation so both measure exactly the same
    quantity. ``recon`` is whatever splats object to render (the detached
    ``recon_d`` during training, the live ``recon`` during validation).

    When ``return_extras=True``, a fifth element — a dict with the rendered
    RGB (``"rendered"``), and, when enabled, the camera-frame rendered normals
    (``"normals_cam"``) and their loss mask (``"normal_mask"``), and the
    rendered depth (``"rendered_depth"``) — is also returned. This lets
    validation compute image/geometry quality metrics without re-rendering.
    """
    v_mats = fd["v_mats"].to(device, non_blocking=True)
    v_K    = fd["v_K"].to(device, non_blocking=True)
    v_gt   = fd["v_gt"].to(device, non_blocking=True)
    v_mask = fd["v_mask"].to(device, non_blocking=True)
    v_h, v_w = fd["v_h"], fd["v_w"]

    normal_loss = torch.zeros((), device=device)
    depth_loss  = torch.zeros((), device=device)

    if use_normals:
        v_gt_normals = fd["v_gt_normals"].to(device, non_blocking=True)
        rendered, rendered_normals = render_view(
            recon, v_mats, v_K, v_w, v_h,
            output_type=output_type, device=device, ret_normals=True,
        )
        photo_loss  = _photometric_loss(rendered, v_gt, v_mask, loss_cfg)
        normals_cam = world_normals_to_camera(rendered_normals, v_mats)
        depth_valid = (v_gt_normals.norm(dim=1, keepdim=True) > 0.5).float()
        if loss_cfg.normal_region == NormalRegion.full:
            normal_mask = depth_valid
        else:
            normal_mask = (v_mask > 0.5).float() * depth_valid
        if loss_cfg.normal_loss_type == NormalLossType.cosine:
            dot = (normals_cam * v_gt_normals).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
            normal_loss = ((1.0 - dot) * normal_mask).mean()
        else:
            normal_loss = F.l1_loss(normals_cam * normal_mask,
                                    v_gt_normals * normal_mask)
        if use_depth:
            _, rendered_depth = render_view(
                recon, v_mats, v_K, v_w, v_h,
                output_type=output_type, device=device, ret_depth=True,
            )
    else:
        if use_depth:
            rendered, rendered_depth = render_view(
                recon, v_mats, v_K, v_w, v_h,
                output_type=output_type, device=device, ret_depth=True,
            )
        else:
            rendered = render_view(recon, v_mats, v_K, v_w, v_h,
                                   output_type=output_type, device=device)
        photo_loss = _photometric_loss(rendered, v_gt, v_mask, loss_cfg)

    if use_depth:
        v_depth      = fd["v_depth"].to(device, non_blocking=True)
        v_depth_mask = fd["v_depth_mask"].to(device, non_blocking=True)
        depth_loss   = _depth_loss(rendered_depth, v_depth, v_depth_mask)

    frame_loss = (photo_loss
                  + loss_cfg.lambda_normal * normal_loss
                  + loss_cfg.lambda_depth * depth_loss)
    if not return_extras:
        return frame_loss, photo_loss, normal_loss, depth_loss

    extras = {"rendered": rendered}
    if use_normals:
        extras["normals_cam"]  = normals_cam
        extras["normal_mask"] = normal_mask
    if use_depth:
        extras["rendered_depth"] = rendered_depth
    return frame_loss, photo_loss, normal_loss, depth_loss, extras


_DEPTH_ERR_KEYS = ("l1_abs", "l2_abs", "l1_rel", "l2_rel")


@torch.no_grad()
def _validate(model, gaussian_cache, val_loader, loss_cfg, output_type,
              use_normals, use_depth, max_points, device, seed=None):
    """Average validation loss/metrics over the held-out sequences (no gradients).

    Frame data (cameras / GT / normals / depth) is computed on the fly from
    the ``val_loader`` batches. Besides the training-objective losses, also
    computes standard image/geometry quality metrics (PSNR, SSIM, LPIPS, and
    — when enabled — depth L1/L2 relative/absolute error and normal angular
    error) for reporting to wandb.

    When ``seed`` is given, the ``max_points`` subsampling is drawn from a
    dedicated generator seeded with it, so every call evaluates the exact
    same subsampled splats — metrics are comparable across epochs and across
    checkpoints.
    """
    model.eval()
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    tot_loss = tot_photo = tot_normal = tot_depth = 0.0
    tot_psnr = tot_ssim = tot_lpips = 0.0
    tot_normal_err = 0.0
    tot_depth_err  = {k: 0.0 for k in _DEPTH_ERR_KEYS}
    n_seq = 0
    for batch in val_loader:
        seq    = _seq_name(batch)
        frames = _frames_from_batch(batch, use_normals, use_depth)
        if not frames:
            continue
        T = len(frames)
        splats = _subsample(_to_device(gaussian_cache[seq], device), max_points, gen)
        recon  = model.reconstruct_splats(splats)
        s_loss = s_photo = s_normal = s_depth = 0.0
        s_psnr = s_ssim = s_lpips = s_normal_err = 0.0
        s_depth_err = {k: 0.0 for k in _DEPTH_ERR_KEYS}
        for fd in frames:
            fl, pl, nl, dl, extras = _frame_losses(
                recon, fd, loss_cfg, output_type, use_normals, use_depth, device,
                return_extras=True)
            s_loss   += fl.item() / T
            s_photo  += pl.item() / T
            s_normal += nl.item() / T
            s_depth  += dl.item() / T

            rendered = extras["rendered"].clamp(0.0, 1.0)
            v_gt     = fd["v_gt"].to(device, non_blocking=True).clamp(0.0, 1.0)
            metric_mask = None
            if loss_cfg.recon_region == NormalRegion.foreground:
                metric_mask = fd["v_mask"].to(device, non_blocking=True)
            s_psnr  += metrics.psnr(rendered, v_gt, metric_mask).item() / T
            s_ssim  += metrics.ssim(rendered, v_gt, metric_mask).item() / T
            s_lpips += metrics.lpips_distance(rendered, v_gt, device, metric_mask).item() / T

            if use_depth:
                v_depth      = fd["v_depth"].to(device, non_blocking=True)
                v_depth_mask = fd["v_depth_mask"].to(device, non_blocking=True)
                derr = metrics.depth_errors(extras["rendered_depth"], v_depth, v_depth_mask)
                for k in _DEPTH_ERR_KEYS:
                    s_depth_err[k] += derr[k].item() / T
            if use_normals:
                v_gt_normals = fd["v_gt_normals"].to(device, non_blocking=True)
                s_normal_err += metrics.normal_angular_error(
                    extras["normals_cam"], v_gt_normals, extras["normal_mask"]).item() / T

        tot_loss   += s_loss
        tot_photo  += s_photo
        tot_normal += s_normal
        tot_depth  += s_depth
        tot_psnr   += s_psnr
        tot_ssim   += s_ssim
        tot_lpips  += s_lpips
        tot_normal_err += s_normal_err
        for k in _DEPTH_ERR_KEYS:
            tot_depth_err[k] += s_depth_err[k]
        n_seq += 1
    if device.type == "cuda":
        # Return the reserve fragmented by ~2k full-frame renders to the
        # driver, so the next training epoch's large contiguous allocations
        # (gsplat isect_tiles) don't OOM on a nearly-full GPU.
        torch.cuda.empty_cache()
    n_seq = max(n_seq, 1)
    result = {
        "loss":   tot_loss   / n_seq,
        "photo":  tot_photo  / n_seq,
        "normal": tot_normal / n_seq,
        "depth":  tot_depth  / n_seq,
        "psnr":   tot_psnr   / n_seq,
        "ssim":   tot_ssim   / n_seq,
        "lpips":  tot_lpips  / n_seq,
    }
    if use_normals:
        result["normal_err"] = tot_normal_err / n_seq
    if use_depth:
        for k in _DEPTH_ERR_KEYS:
            result[f"depth_{k}"] = tot_depth_err[k] / n_seq
    return result


def _build_gaussian_cache(splat_loader, output_type, num_sequences, verbose=False,
                          cache_dir: Path | None = None, shard_size: int = 100):
    """Preload per-sequence splats. ``num_sequences`` caps the count (``None`` = all).

    When ``cache_dir`` is given, converted splats are written to disk in
    shards (see ``DiskGaussianCache``) instead of being kept resident in a
    Python dict, so RAM stays bounded regardless of dataset size. If the
    directory already holds a cache covering the requested sequences (from a
    previous run with the same output_type), it is reused without rebuilding.
    """
    sampler = splat_loader.batch_sampler
    # Splats are per-sequence — only the first frame index is needed.
    sampler._batches = [b[:1] for b in sampler._batches]
    cache: dict[str, GaussianSplats] | DiskGaussianCache
    if cache_dir is not None:
        needed = list(sampler._seq_names)
        if num_sequences is not None:
            needed = needed[:num_sequences]
        cache = DiskGaussianCache(cache_dir, shard_size, output_type=output_type)
        if cache.try_reuse(needed, output_type):
            print(f"  reusing {len(cache)} cached sequences from {cache_dir}")
            return cache
    else:
        cache = {}
    for batch in splat_loader:
        seq = _seq_name(batch)
        raw = batch["sequence_gaussian_splats"][0]
        splats = _cache_splat(raw, output_type)
        if isinstance(cache, DiskGaussianCache):
            cache.add(seq, splats)
        else:
            cache[seq] = splats
        if verbose:
            print(f"  {seq}: {raw.means.shape[0]} points")
        if num_sequences is not None and len(cache) >= num_sequences:
            break
    if isinstance(cache, DiskGaussianCache):
        cache.finalize()
    return cache


class DiskGaussianCache:
    """Gaussian splat cache backed by sharded .pt files on disk.

    Shards multiple sequences per file (rather than one file per sequence)
    since a 6k-sequence run would otherwise create thousands of tiny files.
    Shards are filled sequentially and flushed to disk as soon as they hold
    ``shard_size`` sequences, so peak staging memory is one shard — never the
    whole cache. ``finalize()`` writes an ``index.json`` (sequence → shard
    mapping plus the output_type) so later runs can reuse the shards via
    ``try_reuse()`` instead of rebuilding.

    Random access (``__getitem__``) keeps the most recently loaded shard
    resident, so iterating sequences in index order costs one shard load per
    ``shard_size`` sequences.
    """

    def __init__(self, cache_dir: Path, shard_size: int = 100,
                 output_type: str | None = None):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._shard_size = max(1, shard_size)
        self._output_type = output_type
        self._index: dict[str, int] = {}   # insertion order = shard-major order
        self._staged: dict[str, GaussianSplats] = {}  # current (unflushed) shard
        self._loaded: tuple[int, dict[str, GaussianSplats]] | None = None

    def _shard_path(self, shard_id: int) -> Path:
        return self._dir / f"shard_{shard_id:04d}.pt"

    def _index_path(self) -> Path:
        return self._dir / "index.json"

    def add(self, seq: str, splats: GaussianSplats) -> None:
        """Stage a sequence's splats; the shard is flushed to disk once full."""
        self._index[seq] = len(self._index) // self._shard_size
        self._staged[seq] = splats
        if len(self._staged) >= self._shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._staged:
            return
        shard_id = self._index[next(iter(self._staged))]
        torch.save(self._staged, self._shard_path(shard_id))
        self._staged = {}

    def finalize(self) -> None:
        """Flush the last partial shard and persist the index."""
        self._flush()
        meta = {"output_type": self._output_type,
                "seqs": [[s, i] for s, i in self._index.items()]}
        self._index_path().write_text(json.dumps(meta))

    def try_reuse(self, needed: list[str], output_type: str) -> bool:
        """Adopt an existing on-disk cache if it covers ``needed`` sequences.

        Prefers ``index.json``; falls back to scanning legacy shard files
        written before the index existed (those are assumed to match the
        requested output_type). Returns False — leaving the cache empty, to
        be rebuilt — on any mismatch.
        """
        if self._index_path().exists():
            try:
                meta = json.loads(self._index_path().read_text())
            except (json.JSONDecodeError, OSError):
                return False
            if meta.get("output_type") != output_type:
                return False
            seq_list = [(s, i) for s, i in meta["seqs"]]
        else:
            seq_list = self._scan_legacy_shards(output_type)
            if seq_list is None:
                return False
        index = dict(seq_list)
        needed_set = set(needed)
        if not needed_set <= index.keys():
            return False
        if any(not self._shard_path(i).exists() for i in set(index.values())):
            return False
        # Keep only the needed sequences, preserving shard-major order.
        self._index = {s: i for s, i in seq_list if s in needed_set}
        return True

    def _scan_legacy_shards(self, output_type: str) -> list[tuple[str, int]] | None:
        shard_files = sorted(self._dir.glob("shard_*.pt"))
        if not shard_files:
            return None
        print(f"  found {len(shard_files)} shards without index.json — scanning "
              f"(assuming output_type={output_type})...")
        seq_list: list[tuple[str, int]] = []
        for p in shard_files:
            shard_id = int(p.stem.split("_")[1])
            shard = torch.load(p, weights_only=False)
            seq_list += [(s, shard_id) for s in shard]
        meta = {"output_type": output_type, "seqs": [[s, i] for s, i in seq_list]}
        self._index_path().write_text(json.dumps(meta))
        return seq_list

    def __getitem__(self, seq: str) -> GaussianSplats:
        if seq in self._staged:
            return self._staged[seq]
        shard_id = self._index[seq]
        if self._loaded is None or self._loaded[0] != shard_id:
            self._loaded = (shard_id,
                            torch.load(self._shard_path(shard_id), weights_only=False))
        return self._loaded[1][seq]

    def items(self):
        for seq in self._index:
            yield seq, self[seq]

    def values(self):
        for _, splats in self.items():
            yield splats

    def __iter__(self):
        return iter(self._index)

    def __contains__(self, seq: str) -> bool:
        return seq in self._index

    def __len__(self) -> int:
        return len(self._index)


def _frames_from_batch(batch, use_normals, use_depth) -> list[dict]:
    """Convert one sequence batch into per-frame training dicts (CPU tensors)."""
    _cpu = torch.device("cpu")
    frames = []
    for i in range(batch["image_rgb"].shape[0]):
        fr = _slice_frame(batch, i)
        if use_normals:
            vm, K, gt, mask, h, w, _, normals = \
                get_camera_params(fr, _cpu, ret_normals=True)
            fd: dict = dict(v_mats=vm, v_K=K, v_gt=gt, v_mask=mask,
                            v_h=h, v_w=w, v_gt_normals=normals)
        else:
            vm, K, gt, mask, h, w = get_camera_params(fr, _cpu)
            fd = dict(v_mats=vm, v_K=K, v_gt=gt, v_mask=mask, v_h=h, v_w=w)
        if use_depth:
            dsa = fr["depth_scale_adjustment"]
            dsa = dsa.view(-1)[0].item() if torch.is_tensor(dsa) else float(dsa)
            fd["v_depth"]      = fr["depth_map"].float() * dsa
            fd["v_depth_mask"] = (fr["depth_mask"] > 0.5).float()
        frames.append(fd)
    return frames


def _supervision_views(image_loader, gaussian_cache, n_views) -> dict[str, list[int]]:
    """Map each cached sequence to its evenly-spaced supervision frame indices.

    ``n_views <= 0`` selects every 10th frame of the sequence.
    """
    sampler = image_loader.batch_sampler
    if n_views <= 0:
        return {seq: b[::10]
                for seq, b in zip(sampler._seq_names, sampler._batches)
                if seq in gaussian_cache}
    return {
        seq: b[:: max(1, len(b) // n_views)][:n_views]
        for seq, b in zip(sampler._seq_names, sampler._batches)
        if seq in gaussian_cache
    }


def _set_epoch_batches(image_loader, sup_views, seq_order, views_per_step,
                       generator=None):
    """Point the image loader at one batch per sequence, in ``seq_order``.

    With ``views_per_step`` > 0 and a generator, each batch is a fresh random
    subset of the sequence's supervision views (call once per epoch to
    resample); otherwise batches contain all supervision views. Frames are
    loaded lazily by the DataLoader workers when the loader is iterated —
    nothing is precomputed or cached.
    """
    batches, names = [], []
    for seq in seq_order:
        views = sup_views[seq]
        if generator is not None and 0 < views_per_step < len(views):
            picks = torch.randperm(len(views), generator=generator)[:views_per_step]
            batches.append([views[i] for i in sorted(picks.tolist())])
        else:
            batches.append(list(views))
        names.append(seq)
    sampler = image_loader.batch_sampler
    sampler._batches = batches
    sampler._seq_names = names


def build_val_set(cfg, dm, output_type, cache_dir: Path):
    """Validation gaussian cache + deterministic image loader over the val split.

    Single source of truth for what "the validation set" means, shared by the
    training loop and eval_val_ckpts.py so both measure the identical set:
    the "val" split filtered by ``data.max_splats``, capped to
    ``data.num_val_sequences`` sequences (0 = the full split), with
    ``data.num_val_views`` fixed evenly-spaced frames per sequence
    (0 = every frame).

    Returns ``(gaussian_cache, loader, n_frames, n_seqs)``; ``loader`` is
    ``None`` when the split yields no usable sequences.
    """
    max_splats   = cfg.data.max_splats or None
    num_val_seqs = cfg.data.num_val_sequences or None
    n_val_views  = cfg.data.num_val_views

    val_gaussian_cache = _build_gaussian_cache(
        dm.splat_only_loader(max_splats=max_splats, subsets=("val",)),
        output_type, num_val_seqs, cache_dir=cache_dir)
    val_loader    = dm.image_only_loader(subsets=("val",), shuffle=False)
    val_sup_views = _supervision_views(val_loader, val_gaussian_cache, n_val_views)
    val_seq_order = [s for s in val_gaussian_cache if s in val_sup_views]
    if not val_seq_order:
        return val_gaussian_cache, None, 0, 0
    # Fixed, evenly-spaced views so metrics are comparable across epochs.
    _set_epoch_batches(val_loader, val_sup_views, val_seq_order, 0)
    n_frames = sum(len(val_sup_views[s]) for s in val_seq_order)
    return val_gaussian_cache, val_loader, n_frames, len(val_seq_order)


def _init_wandb(cfg, output_dir: Path):
    if cfg.run.wandb_mode == WandbMode.disabled:
        return None
    return wandb.init(
        project=cfg.run.wandb_project,
        name=cfg.run.name,
        mode=cfg.run.wandb_mode.value,
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        dir=str(output_dir),
    )


def _seq_name(batch) -> str:
    n = batch["sequence_name"]
    return n[0] if isinstance(n, (list, tuple)) else n


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = parse_cli()
    torch.manual_seed(cfg.run.seed)

    device     = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    print(f"Output dir: {output_dir}  |  device: {device}")

    wandb_run = _init_wandb(cfg, output_dir)
    if wandb_run is not None:
        print(f"wandb: project={cfg.run.wandb_project} run={wandb_run.name}")
        art = wandb.Artifact(f"{cfg.run.name}-config", type="config")
        art.add_file(str(output_dir / "config.yaml"))
        wandb.log_artifact(art)

    output_type = cfg.model.output_type.value   # "3dgs" | "2dgs" | "linprim"
    use_normals = cfg.train.loss.lambda_normal > 0
    use_depth   = cfg.train.loss.lambda_depth  > 0
    if use_normals and output_type != "2dgs":
        raise ValueError(
            "loss.lambda_normal > 0 requires model.output_type=gs2d "
            "(only the 2dgs rasterizer produces surface normals)"
        )
    print(f"Output type: {output_type}  |  normal supervision: {use_normals}  |  depth supervision: {use_depth}")

    # ── data module ───────────────────────────────────────────────────────────
    dm = UCO3DDataModule(
        subset_lists_filename=cfg.data.subset_lists_filename,
        dataset_root=cfg.data.dataset_root,
        num_workers=cfg.data.num_workers,
        image_height=cfg.data.image_height,
        image_width=cfg.data.image_width,
        load_masks=cfg.data.load_masks,
        load_depths=use_normals or use_depth,
        load_gaussian_splats=True,
        pick_sequences=list(cfg.data.pick_sequences),
    )

    # ── phase 1: build gaussian cache (splats only, images skipped) ───────────
    max_splats    = cfg.data.max_splats    or None
    max_points    = cfg.data.max_points            # 0 = no subsampling
    num_sequences = cfg.data.num_sequences or None
    n_views       = cfg.data.num_supervision_views

    _gaussian_cache_dir = output_dir / "gaussian_cache"
    print(f"Preloading Gaussian cache (disk-backed → {_gaussian_cache_dir})...")
    gaussian_cache = _build_gaussian_cache(
        dm.splat_only_loader(max_splats=max_splats),
        output_type, num_sequences, verbose=True, cache_dir=_gaussian_cache_dir)
    print(f"Cached {len(gaussian_cache)} sequences.\n")

    # ── model & optimizer ─────────────────────────────────────────────────────
    model     = build_model(cfg.model).to(device)
    optimizer = _build_optimizer(model.parameters(), cfg.train)
    scheduler = _build_scheduler(optimizer, cfg.train)
    print(f"Model: {model.__class__.__name__}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    start_epoch = 0
    if cfg.train.resume:
        start_epoch = _resume_from_checkpoint(
            Path(cfg.train.resume), cfg.train.resume_epoch,
            model, optimizer, scheduler, device)

    # ── phase 2: frame data loaded on the fly by the image loader ────────────
    # Batches follow the gaussian cache's shard-major order so that random
    # splat access (DiskGaussianCache.__getitem__) reuses the resident shard.
    image_loader = dm.image_only_loader(shuffle=False)
    sup_views    = _supervision_views(image_loader, gaussian_cache, n_views)
    seq_order    = [s for s in gaussian_cache if s in sup_views]
    n_step_views = cfg.train.num_views_per_step
    print(f"Frame data: on the fly, {len(seq_order)} sequences × up to {n_views} "
          f"supervision views ({n_step_views or 'all'} views per step).\n")

    # First supervision view of the first few sequences, kept resident for the
    # periodic reconstruction visualizations.
    vis_seqs = seq_order[: cfg.train.max_log_sequences]
    vis_frames = {}
    for seq in vis_seqs:
        _b = uco3d_collate([image_loader.dataset[sup_views[seq][0]]])
        vis_frames[seq] = _frames_from_batch(_b, use_normals, use_depth)[0]

    # ── validation (full held-out "val" split, on-the-fly frames) ────────────
    val_gaussian_cache: dict[str, GaussianSplats] | DiskGaussianCache = {}
    val_loader = None
    if cfg.train.val_every > 0:
        print(f"Preloading validation cache "
              f"({cfg.data.num_val_sequences or 'all'} val sequences)...")
        val_gaussian_cache, val_loader, n_val_frames, n_val_seqs = build_val_set(
            cfg, dm, output_type, output_dir / "gaussian_cache_val")
        if val_loader is not None:
            print(f"Validation: {n_val_frames} frames across {n_val_seqs} sequences.\n")
        else:
            print("WARNING: no validation sequences found in the 'val' split; "
                  "validation disabled.\n")

    # ── profiler setup ────────────────────────────────────────────────────────
    _profile_steps = cfg.train.profile_steps
    _profiler = None
    if _profile_steps > 0:
        _activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            _activities.append(torch.profiler.ProfilerActivity.CUDA)
        _profiler = torch.profiler.profile(
            activities=_activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=_profile_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir / "profiler")),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        _profiler.__enter__()
        print(f"Profiling: recording {_profile_steps} steps after 1 wait + 1 warmup. "
              f"Trace → {output_dir / 'profiler'}\n")

    # ── training loop ─────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            model.train()
            total_loss = total_photo = total_normal = total_depth = 0.0
            total_t_forward = total_t_render = total_t_backward = 0.0
            n_steps = 0
            t_epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            # Resample each sequence's step views, then let the DataLoader
            # workers prefetch them while the GPU trains. Batches stay in
            # gaussian-shard order so gaussian_cache[seq] hits the resident
            # shard — see DiskGaussianCache.
            _rng = torch.Generator()
            _rng.manual_seed(cfg.run.seed + epoch)
            _set_epoch_batches(image_loader, sup_views, seq_order, n_step_views, _rng)
            for batch in image_loader:
                seq         = _seq_name(batch)
                step_frames = _frames_from_batch(batch, use_normals, use_depth)
                if not step_frames:
                    continue
                T = len(step_frames)

                optimizer.zero_grad(set_to_none=True)

                _sync(device)
                _t0 = time.perf_counter()
                splats = _subsample(_to_device(gaussian_cache[seq], device), max_points)
                recon  = model.reconstruct_splats(splats)
                _sync(device)
                total_t_forward += time.perf_counter() - _t0

                # Detach recon from the network graph so each frame's render
                # backward only traverses the rasterizer, not the full network.
                # We accumulate render→recon gradients and do one network
                # backward at the end.
                _recon_src = [recon.means, recon.sh0, recon.scales,
                              recon.quats, recon.opacities]
                _recon_d   = [t.detach().requires_grad_(True) for t in _recon_src]
                recon_d = GaussianSplats(
                    means=_recon_d[0], sh0=_recon_d[1], scales=_recon_d[2],
                    quats=_recon_d[3], opacities=_recon_d[4],
                    shN=None, fg_mask=None,
                )
                _recon_grads = [torch.zeros_like(t) for t in _recon_d]

                seq_photo  = seq_normal = seq_depth = seq_loss_val = 0.0

                _sync(device)
                _t0 = time.perf_counter()
                for fd in step_frames:
                    frame_loss, photo_loss, normal_loss, depth_loss = _frame_losses(
                        recon_d, fd, cfg.train.loss, output_type,
                        use_normals, use_depth, device,
                    )
                    seq_photo    += photo_loss.item()
                    seq_normal   += normal_loss.item()
                    seq_depth    += depth_loss.item()
                    seq_loss_val += frame_loss.item() / T
                    # Backward through render only; frees the render graph.
                    (frame_loss / T).backward()
                    for j, rd in enumerate(_recon_d):
                        if rd.grad is not None:
                            _recon_grads[j].add_(rd.grad)
                            rd.grad = None

                _sync(device)
                total_t_render += time.perf_counter() - _t0

                # Single backward through the network using accumulated grads.
                _t0 = time.perf_counter()
                _active = [(s, g) for s, g in zip(_recon_src, _recon_grads)
                           if s.requires_grad]
                if _active:
                    torch.autograd.backward(
                        [s for s, _ in _active], [g for _, g in _active]
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=cfg.train.grad_clip)
                optimizer.step()
                _sync(device)
                total_t_backward += time.perf_counter() - _t0

                total_loss   += seq_loss_val
                total_photo  += seq_photo  / T
                total_normal += seq_normal / T
                total_depth  += seq_depth  / T
                n_steps      += 1
                if _profiler is not None:
                    _profiler.step()

            epoch_time = time.perf_counter() - t_epoch_start
            n_steps    = max(n_steps, 1)
            avg        = total_loss   / n_steps
            avg_photo  = total_photo  / n_steps
            avg_normal = total_normal / n_steps
            avg_depth  = total_depth  / n_steps
            avg_t_forward  = total_t_forward  / n_steps
            avg_t_render   = total_t_render   / n_steps
            avg_t_backward = total_t_backward / n_steps

            if scheduler is not None:
                scheduler.step(avg)

            wandb_log: dict = {
                "train/loss": avg, "train/photo_loss": avg_photo,
                "train/lr": optimizer.param_groups[0]["lr"],
                "time/epoch_s": epoch_time,
                "time/forward_ms": avg_t_forward * 1e3,
                "time/render_ms": avg_t_render * 1e3,
                "time/backward_ms": avg_t_backward * 1e3,
            }
            if use_normals:
                wandb_log["train/normal_loss"] = avg_normal
            if use_depth:
                wandb_log["train/depth_loss"] = avg_depth

            # ── validation pass (held-out sequences, no gradients) ──────────
            val_metrics = None
            if val_loader is not None and epoch % cfg.train.val_every == 0:
                val_metrics = _validate(
                    model, val_gaussian_cache, val_loader, cfg.train.loss,
                    output_type, use_normals, use_depth, max_points, device,
                    seed=cfg.run.seed)
                wandb_log["val/loss"]       = val_metrics["loss"]
                wandb_log["val/photo_loss"] = val_metrics["photo"]
                if use_normals:
                    wandb_log["val/normal_loss"] = val_metrics["normal"]
                if use_depth:
                    wandb_log["val/depth_loss"] = val_metrics["depth"]
                wandb_log["val/psnr"]  = val_metrics["psnr"]
                wandb_log["val/ssim"]  = val_metrics["ssim"]
                wandb_log["val/lpips"] = val_metrics["lpips"]
                if use_normals:
                    wandb_log["val/normal_err_deg"] = val_metrics["normal_err"]
                if use_depth:
                    for k in _DEPTH_ERR_KEYS:
                        wandb_log[f"val/depth_{k}"] = val_metrics[f"depth_{k}"]

            if device.type == "cuda":
                wandb_log["mem/peak_gpu_gb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3

            if epoch % cfg.train.log_every == 0:
                parts = [f"Epoch {epoch:05d}/{cfg.train.epochs}", f"Loss={avg:.6f}"]
                if scheduler is not None:
                    parts.append(f"lr={optimizer.param_groups[0]['lr']:.2e}")
                if val_metrics is not None:
                    parts.append(f"ValLoss={val_metrics['loss']:.6f}")
                    parts.append(f"ValPSNR={val_metrics['psnr']:.2f}")
                if use_normals:
                    parts += [f"Photo={avg_photo:.6f}", f"Normal={avg_normal:.6f}"]
                if use_depth:
                    parts.append(f"Depth={avg_depth:.6f}")
                parts.append(
                    f"t=[fwd={avg_t_forward*1e3:.1f}ms "
                    f"rnd={avg_t_render*1e3:.1f}ms "
                    f"bwd={avg_t_backward*1e3:.1f}ms "
                    f"epoch={epoch_time:.1f}s]"
                )
                print(" | ".join(parts))

                model.eval()
                for si, seq in enumerate(vis_seqs):
                    with torch.no_grad():
                        fd0 = vis_frames[seq]
                        v0_mats = fd0["v_mats"].to(device)
                        v0_K    = fd0["v_K"].to(device)
                        v0_gt   = fd0["v_gt"].to(device)
                        v0_h, v0_w = fd0["v_h"], fd0["v_w"]
                        if use_normals:
                            v0_gt_normals = fd0["v_gt_normals"].to(device)
                        if use_depth:
                            v0_depth = fd0["v_depth"].to(device)

                        splats_vis = _subsample(_to_device(gaussian_cache[seq], device), max_points)
                        eval_recon = model.reconstruct_splats(splats_vis)
                        image_path = output_dir / f"epoch_{epoch:05d}_seq{si:02d}.png"

                        # The linprim cache collapses scales to a single dist
                        # (N, 1), which the 3DGS rasterizer can't consume, so
                        # render the un-refined reference through the linprim
                        # rasterizer instead.
                        ref_output_type = "linprim" if output_type == "linprim" else "3dgs"
                        if use_depth:
                            ref_render, ref_depth = render_view(
                                splats_vis, v0_mats, v0_K, v0_w, v0_h,
                                output_type=ref_output_type, device=device, ret_depth=True,
                            )
                        else:
                            ref_render = render_view(splats_vis, v0_mats, v0_K, v0_w, v0_h,
                                                     output_type=ref_output_type, device=device)

                        if use_normals:
                            eval_rgb, eval_normals = render_view(
                                eval_recon, v0_mats, v0_K, v0_w, v0_h,
                                output_type=output_type, device=device, ret_normals=True,
                            )
                            eval_normals = world_normals_to_camera(eval_normals, v0_mats)
                            _, ref_normals = render_view(
                                splats_vis, v0_mats, v0_K, v0_w, v0_h,
                                output_type="2dgs", device=device, ret_normals=True,
                            )
                            ref_normals = world_normals_to_camera(ref_normals, v0_mats)
                            if use_depth:
                                _, eval_depth = render_view(
                                    eval_recon, v0_mats, v0_K, v0_w, v0_h,
                                    output_type=output_type, device=device, ret_depth=True,
                                )
                        elif use_depth:
                            eval_rgb, eval_depth = render_view(
                                eval_recon, v0_mats, v0_K, v0_w, v0_h,
                                output_type=output_type, device=device, ret_depth=True,
                            )
                        else:
                            eval_rgb = render_view(eval_recon, v0_mats, v0_K, v0_w, v0_h,
                                                   output_type=output_type, device=device)

                        def _nrm(n): return (n * 0.5 + 0.5).clamp(0, 1)
                        def _dep(d):
                            d = d.squeeze(1)
                            valid = d > 0
                            lo = d[valid].min() if valid.any() else d.min()
                            return ((d - lo) / (d.max() - lo + 1e-8)).clamp(0, 1).unsqueeze(-1).expand(-1, -1, -1, 3)

                        rows = [torch.cat([v0_gt.permute(0,2,3,1),
                                           ref_render.permute(0,2,3,1),
                                           eval_rgb.permute(0,2,3,1)], dim=2)]
                        labels = ["GT", f"Reference ({ref_output_type})", f"Recon @ {epoch}"]

                        if use_normals:
                            rows.append(torch.cat([_nrm(v0_gt_normals).permute(0,2,3,1),
                                                   _nrm(ref_normals).permute(0,2,3,1),
                                                   _nrm(eval_normals).permute(0,2,3,1)], dim=2))
                            labels += ["GT normals", "Reference normals", f"Recon normals @ {epoch}"]

                        if use_depth:
                            rows.append(torch.cat([_dep(v0_depth),
                                                   _dep(ref_depth),
                                                   _dep(eval_depth)], dim=2))
                            labels += ["GT depth", "Reference depth", f"Recon depth @ {epoch}"]

                        SplatRenderVisualizer.save(
                            str(image_path),
                            render_colors=torch.cat(rows, dim=1),
                            render_alphas=torch.ones(1, v0_h * len(rows), v0_w * 3, 1, device=device),
                            labels=labels,
                            columns=3,
                        )

                        if wandb_run is not None:
                            wandb_log[f"recon/seq{si:02d}"] = wandb.Image(str(image_path))

            if wandb_run is not None:
                wandb.log(wandb_log, step=epoch)

            if cfg.train.ckpt_every > 0 and epoch > 0 and epoch % cfg.train.ckpt_every == 0:
                ckpt_path = output_dir / f"ckpt_{epoch:05d}.pt"
                torch.save(model.state_dict(), ckpt_path)
                _save_trainstate(ckpt_path, optimizer, scheduler, epoch)

        torch.save(model.state_dict(), output_dir / "ckpt_final.pt")
        _save_trainstate(output_dir / "ckpt_final.pt", optimizer, scheduler,
                         cfg.train.epochs - 1)
        print(f"Done. Final checkpoint: {output_dir / 'ckpt_final.pt'}")
        if wandb_run is not None:
            art = wandb.Artifact(f"{cfg.run.name}-ckpt-final", type="model")
            art.add_file(str(output_dir / "ckpt_final.pt"))
            wandb.log_artifact(art)
    finally:
        if _profiler is not None:
            _profiler.__exit__(None, None, None)
            print("\n=== Profiler: top ops by CUDA time ===")
            print(_profiler.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
