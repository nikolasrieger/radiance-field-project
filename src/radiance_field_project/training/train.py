"""Config-driven training entry-point.

Usage
-----
    uv run training --config configs/pointnet2.yaml
    uv run training --config configs/minipointnet.yaml run.name=my_run train.lr=5e-4

Everything except --config is forwarded to OmegaConf as dotlist overrides.

Training is two-phase:
  1. Gaussian cache  — splat_only_loader preloads all sequence splats to CPU once.
  2. Training loop   — image_only_loader (SequenceBatchSampler) drives each epoch.
"""
from __future__ import annotations

from pathlib import Path
import time

import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.optim import Adam, AdamW
from uco3d import GaussianSplats
from uco3d.dataset_utils.data_types import Cameras

from radiance_field_project.config import parse_cli
from radiance_field_project.config.schema import LossType, NormalLossType, NormalRegion, Optimizer, OutputType, WandbMode
from radiance_field_project.data.datamodule import UCO3DDataModule
from radiance_field_project.models.model_factory import build_model
from radiance_field_project.utils.gaussian_ops import to_2dgs_canonical
from radiance_field_project.utils.losses import reconstruction_loss, reconstruction_loss_l1
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


def _subsample(splats: GaussianSplats, max_points: int) -> GaussianSplats:
    if max_points <= 0 or splats.means.shape[0] <= max_points:
        return splats
    perm = torch.randperm(splats.means.shape[0], device=splats.means.device)[:max_points]
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


def _frame_losses(recon, fd, loss_cfg, output_type, use_normals, use_depth, device):
    """Per-frame losses for ``recon`` against cached frame data ``fd``.

    Returns ``(frame_loss, photo_loss, normal_loss, depth_loss)`` as tensors;
    the normal/depth terms are zero tensors when their supervision is disabled.
    Shared by the training step and validation so both measure exactly the same
    quantity. ``recon`` is whatever splats object to render (the detached
    ``recon_d`` during training, the live ``recon`` during validation).
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
    return frame_loss, photo_loss, normal_loss, depth_loss


@torch.no_grad()
def _validate(model, gaussian_cache, frame_cache, loss_cfg, output_type,
              use_normals, use_depth, max_points, device):
    """Average validation loss over the held-out sequences (no gradients)."""
    model.eval()
    tot_loss = tot_photo = tot_normal = tot_depth = 0.0
    n_seq = 0
    for seq, splats_cpu in gaussian_cache.items():
        frames = frame_cache.get(seq)
        if not frames:
            continue
        T = len(frames)
        splats = _subsample(_to_device(splats_cpu, device), max_points)
        recon  = model.reconstruct_splats(splats)
        s_loss = s_photo = s_normal = s_depth = 0.0
        for fd in frames:
            fl, pl, nl, dl = _frame_losses(
                recon, fd, loss_cfg, output_type, use_normals, use_depth, device)
            s_loss   += fl.item() / T
            s_photo  += pl.item() / T
            s_normal += nl.item() / T
            s_depth  += dl.item() / T
        tot_loss   += s_loss
        tot_photo  += s_photo
        tot_normal += s_normal
        tot_depth  += s_depth
        n_seq += 1
    n_seq = max(n_seq, 1)
    return {
        "loss":   tot_loss   / n_seq,
        "photo":  tot_photo  / n_seq,
        "normal": tot_normal / n_seq,
        "depth":  tot_depth  / n_seq,
    }


def _build_gaussian_cache(splat_loader, output_type, num_sequences, verbose=False):
    """Preload per-sequence splats to CPU. ``num_sequences`` caps the count
    (``None`` = all)."""
    # Splats are per-sequence — only the first frame index is needed.
    splat_loader.batch_sampler._batches = [
        b[:1] for b in splat_loader.batch_sampler._batches
    ]
    cache: dict[str, GaussianSplats] = {}
    for batch in splat_loader:
        seq = _seq_name(batch)
        raw = batch["sequence_gaussian_splats"][0]
        cache[seq] = _cache_splat(raw, output_type)
        if verbose:
            print(f"  {seq}: {raw.means.shape[0]} points")
        if num_sequences is not None and len(cache) >= num_sequences:
            break
    return cache


class DiskFrameCache:
    """Frame cache backed by per-sequence .pt files on disk to avoid OOM on large datasets."""

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._keys: list[str] = []

    def _path(self, seq: str) -> Path:
        return self._dir / f"{seq}.pt"

    def save(self, seq: str, frames: list[dict]) -> None:
        torch.save(frames, self._path(seq))
        self._keys.append(seq)

    def get(self, seq: str) -> list[dict] | None:
        p = self._path(seq)
        return torch.load(p, weights_only=False) if p.exists() else None

    def __getitem__(self, seq: str) -> list[dict]:
        frames = self.get(seq)
        if frames is None:
            raise KeyError(seq)
        return frames

    def items(self):
        for seq in self._keys:
            yield seq, torch.load(self._path(seq), weights_only=False)

    def values(self):
        for seq in self._keys:
            yield torch.load(self._path(seq), weights_only=False)

    def __len__(self) -> int:
        return len(self._keys)


def _build_frame_cache(image_loader, gaussian_cache, n_views, use_normals, use_depth,
                       cache_dir: Path | None = None):
    """Precompute per-sequence frame data (cameras / GT / normals / depth).

    When cache_dir is provided frames are saved to disk one sequence at a time
    so memory stays bounded regardless of dataset size.
    """
    seq_names = list(image_loader.dataset.sequence_names())
    image_loader.batch_sampler._batches = [
        b[:: max(1, len(b) // n_views)][:n_views]
        for seq, b in zip(seq_names, image_loader.batch_sampler._batches)
        if seq in gaussian_cache
    ]
    _cpu = torch.device("cpu")
    frame_cache: dict[str, list[dict]] | DiskFrameCache = (
        DiskFrameCache(cache_dir) if cache_dir is not None else {}
    )
    for _batch in image_loader:
        _seq = _seq_name(_batch)
        _frames = []
        for _i in range(_batch["image_rgb"].shape[0]):
            _fr = _slice_frame(_batch, _i)
            if use_normals:
                _vm, _K, _gt, _mask, _h, _w, _, _normals = \
                    get_camera_params(_fr, _cpu, ret_normals=True)
                _fd: dict = dict(v_mats=_vm, v_K=_K, v_gt=_gt, v_mask=_mask,
                                 v_h=_h, v_w=_w, v_gt_normals=_normals)
            else:
                _vm, _K, _gt, _mask, _h, _w = get_camera_params(_fr, _cpu)
                _fd = dict(v_mats=_vm, v_K=_K, v_gt=_gt, v_mask=_mask, v_h=_h, v_w=_w)
            if use_depth:
                _dsa = _fr["depth_scale_adjustment"]
                _dsa = _dsa.view(-1)[0].item() if torch.is_tensor(_dsa) else float(_dsa)
                _fd["v_depth"]      = _fr["depth_map"].float() * _dsa
                _fd["v_depth_mask"] = (_fr["depth_mask"] > 0.5).float()
            _frames.append(_fd)
        if isinstance(frame_cache, DiskFrameCache):
            frame_cache.save(_seq, _frames)
        else:
            frame_cache[_seq] = _frames
    return frame_cache


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
    num_val_seqs  = cfg.data.num_val_sequences or None
    n_val_views   = cfg.data.num_val_views or n_views

    print("Preloading Gaussian cache...")
    gaussian_cache = _build_gaussian_cache(
        dm.splat_only_loader(max_splats=max_splats),
        output_type, num_sequences, verbose=True)
    print(f"Cached {len(gaussian_cache)} sequences.\n")

    # ── model & optimizer ─────────────────────────────────────────────────────
    model     = build_model(cfg.model).to(device)
    optimizer = _build_optimizer(model.parameters(), cfg.train)
    print(f"Model: {model.__class__.__name__}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    # ── phase 2/3: precompute frame data cache (CPU) ──────────────────────────
    _frame_cache_dir = output_dir / "frame_cache"
    print(f"Precomputing frame data cache (disk-backed → {_frame_cache_dir})...")
    frame_cache = _build_frame_cache(
        dm.image_only_loader(), gaussian_cache, n_views, use_normals, use_depth,
        cache_dir=_frame_cache_dir)
    n_cached = sum(len(v) for v in frame_cache.values())
    print(f"Frame cache: {n_cached} frames across {len(frame_cache)} sequences.\n")

    # ── validation caches (held-out "val" split) ─────────────────────────────
    val_gaussian_cache: dict[str, GaussianSplats] = {}
    val_frame_cache: dict[str, list[dict]] = {}
    if num_val_seqs is not None:
        print(f"Preloading validation cache ({cfg.data.num_val_sequences} sequences)...")
        val_gaussian_cache = _build_gaussian_cache(
            dm.splat_only_loader(max_splats=max_splats, subsets=("val",)),
            output_type, num_val_seqs)
        val_frame_cache = _build_frame_cache(
            dm.image_only_loader(subsets=("val",), shuffle=False),
            val_gaussian_cache, n_val_views, use_normals, use_depth,
            cache_dir=output_dir / "frame_cache_val")
        n_val = sum(len(v) for v in val_frame_cache.values())
        if val_frame_cache:
            print(f"Validation cache: {n_val} frames across {len(val_frame_cache)} sequences.\n")
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
        for epoch in range(cfg.train.epochs):
            model.train()
            total_loss = total_photo = total_normal = total_depth = 0.0
            total_t_forward = total_t_render = total_t_backward = 0.0
            n_steps = 0
            t_epoch_start = time.perf_counter()

            for seq, cached_frames in frame_cache.items():
                splats_cpu = gaussian_cache[seq]
                n_step = cfg.train.num_views_per_step
                if n_step > 0 and n_step < len(cached_frames):
                    idxs = torch.randperm(len(cached_frames))[:n_step].tolist()
                    step_frames = [cached_frames[i] for i in idxs]
                else:
                    step_frames = cached_frames
                T = len(step_frames)

                optimizer.zero_grad(set_to_none=True)

                _sync(device)
                _t0 = time.perf_counter()
                splats = _subsample(_to_device(splats_cpu, device), max_points)
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

            wandb_log: dict = {
                "train/loss": avg, "train/photo_loss": avg_photo,
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
            if val_frame_cache and epoch % cfg.train.val_every == 0:
                val_metrics = _validate(
                    model, val_gaussian_cache, val_frame_cache, cfg.train.loss,
                    output_type, use_normals, use_depth, max_points, device)
                wandb_log["val/loss"]       = val_metrics["loss"]
                wandb_log["val/photo_loss"] = val_metrics["photo"]
                if use_normals:
                    wandb_log["val/normal_loss"] = val_metrics["normal"]
                if use_depth:
                    wandb_log["val/depth_loss"] = val_metrics["depth"]

            if epoch % cfg.train.log_every == 0:
                parts = [f"Epoch {epoch:05d}/{cfg.train.epochs}", f"Loss={avg:.6f}"]
                if val_metrics is not None:
                    parts.append(f"ValLoss={val_metrics['loss']:.6f}")
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
                for si, (seq, splats_cpu) in enumerate(gaussian_cache.items()):
                    if si >= cfg.train.max_log_sequences:
                        break
                    with torch.no_grad():
                        fd0 = frame_cache[seq][0]
                        v0_mats = fd0["v_mats"].to(device)
                        v0_K    = fd0["v_K"].to(device)
                        v0_gt   = fd0["v_gt"].to(device)
                        v0_h, v0_w = fd0["v_h"], fd0["v_w"]
                        if use_normals:
                            v0_gt_normals = fd0["v_gt_normals"].to(device)
                        if use_depth:
                            v0_depth = fd0["v_depth"].to(device)

                        splats_vis = _subsample(_to_device(splats_cpu, device), max_points)
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
                torch.save(model.state_dict(), output_dir / f"ckpt_{epoch:05d}.pt")

        torch.save(model.state_dict(), output_dir / "ckpt_final.pt")
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
