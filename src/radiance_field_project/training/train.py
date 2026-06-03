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
from radiance_field_project.utils.losses import reconstruction_loss
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
        "fg_probability":         batch["fg_probability"][i : i + 1],
        "camera":                 _slice_camera(batch["camera"], i),
        "depth_scale_adjustment": dsa[i : i + 1] if torch.is_tensor(dsa) else dsa,
    }
    if "depth_map" in batch:
        frame["depth_map"]  = batch["depth_map"][i : i + 1]
        frame["depth_mask"] = batch["depth_mask"][i : i + 1]
    return frame


# ── gaussian cache helpers ────────────────────────────────────────────────────

def _cache_splat(raw: GaussianSplats, is_2dgs: bool) -> GaussianSplats:
    """Convert a raw UCO3D GaussianSplats to a CPU float tensor, optionally
    canonicalizing for 2DGS (thin axis → position 2)."""
    means     = raw.means.float().cpu()
    sh0       = raw.sh0.float().cpu()
    scales    = raw.scales.float().cpu()
    quats     = raw.quats.float().cpu()
    opacities = raw.opacities.float().cpu()

    if is_2dgs:
        scales2d, quats = to_2dgs_canonical(scales, quats)
        thin   = scales.min(dim=-1, keepdim=True).values
        scales = torch.cat([scales2d, thin], dim=-1)  # (N, 3) canonical

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
    if mask is not None:
        binary = (mask > 0.5).float()
        return ((rendered - gt_image).abs() * binary).sum() / binary.sum().clamp(min=1.0)
    return (rendered - gt_image).abs().mean()


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

    output_type = cfg.model.output_type.value   # "3dgs" | "2dgs"
    is_2dgs     = output_type == "2dgs"
    use_normals = cfg.train.loss.lambda_normal > 0
    use_depth   = cfg.train.loss.lambda_depth  > 0
    if use_normals and not is_2dgs:
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
        load_masks=True,
        load_depths=use_normals or use_depth,
        load_gaussian_splats=True,
        pick_sequences=list(cfg.data.pick_sequences),
    )

    # ── phase 1: build gaussian cache (splats only, images skipped) ───────────
    max_splats    = cfg.data.max_splats    or None
    max_points    = cfg.data.max_points            # 0 = no subsampling
    num_sequences = cfg.data.num_sequences or None

    print("Preloading Gaussian cache...")
    splat_loader = dm.splat_only_loader(max_splats=max_splats)
    # Splats are per-sequence — only the first frame index is needed.
    splat_loader.batch_sampler._batches = [
        b[:1] for b in splat_loader.batch_sampler._batches
    ]

    gaussian_cache: dict[str, GaussianSplats] = {}
    for batch in splat_loader:
        seq = _seq_name(batch)
        raw = batch["sequence_gaussian_splats"][0]
        gaussian_cache[seq] = _cache_splat(raw, is_2dgs)
        print(f"  {seq}: {raw.means.shape[0]} points")
        if num_sequences is not None and len(gaussian_cache) >= num_sequences:
            break
    print(f"Cached {len(gaussian_cache)} sequences.\n")

    # ── phase 2: image loader (SequenceBatchSampler), filtered to cache ───────
    n_views      = cfg.data.num_supervision_views
    image_loader = dm.image_only_loader()
    seq_names    = list(image_loader.dataset.sequence_names())
    image_loader.batch_sampler._batches = [
        b[:: max(1, len(b) // n_views)][:n_views]
        for seq, b in zip(seq_names, image_loader.batch_sampler._batches)
        if seq in gaussian_cache
    ]
    print(f"Image loader: {len(image_loader)} sequences, up to {n_views} frames each.\n")

    # ── model & optimizer ─────────────────────────────────────────────────────
    model     = build_model(cfg.model).to(device)
    optimizer = _build_optimizer(model.parameters(), cfg.train)
    print(f"Model: {model.__class__.__name__}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    # ── phase 3: precompute frame data cache (CPU) ───────────────────────────
    print("Precomputing frame data cache...")
    _cpu = torch.device("cpu")
    frame_cache: dict[str, list[dict]] = {}
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
        frame_cache[_seq] = _frames
    n_cached = sum(len(v) for v in frame_cache.values())
    print(f"Frame cache: {n_cached} frames across {len(frame_cache)} sequences.\n")

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
                T          = len(cached_frames)

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
                for fd in cached_frames:
                    v_mats = fd["v_mats"].to(device, non_blocking=True)
                    v_K    = fd["v_K"].to(device, non_blocking=True)
                    v_gt   = fd["v_gt"].to(device, non_blocking=True)
                    v_mask = fd["v_mask"].to(device, non_blocking=True)
                    v_h, v_w = fd["v_h"], fd["v_w"]
                    if use_normals:
                        v_gt_normals = fd["v_gt_normals"].to(device, non_blocking=True)
                    if use_depth:
                        v_depth      = fd["v_depth"].to(device, non_blocking=True)
                        v_depth_mask = fd["v_depth_mask"].to(device, non_blocking=True)

                    if use_normals:
                        rendered, rendered_normals = render_view(
                            recon_d, v_mats, v_K, v_w, v_h,
                            output_type=output_type, device=device, ret_normals=True,
                        )
                        photo_loss  = _photometric_loss(rendered, v_gt, v_mask, cfg.train.loss)
                        normals_cam = world_normals_to_camera(rendered_normals, v_mats)
                        depth_valid = (v_gt_normals.norm(dim=1, keepdim=True) > 0.5).float()
                        if cfg.train.loss.normal_region == NormalRegion.full:
                            normal_mask = depth_valid
                        else:
                            normal_mask = (v_mask > 0.5).float() * depth_valid
                        if cfg.train.loss.normal_loss_type == NormalLossType.cosine:
                            dot = (normals_cam * v_gt_normals).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
                            normal_loss = ((1.0 - dot) * normal_mask).mean()
                        else:
                            normal_loss = F.l1_loss(normals_cam * normal_mask,
                                                    v_gt_normals * normal_mask)
                        frame_loss  = photo_loss + cfg.train.loss.lambda_normal * normal_loss
                        seq_normal += normal_loss.item()
                        if use_depth:
                            _, rendered_depth = render_view(
                                recon_d, v_mats, v_K, v_w, v_h,
                                output_type=output_type, device=device, ret_depth=True,
                            )
                            depth_loss  = _depth_loss(rendered_depth, v_depth, v_depth_mask)
                            frame_loss  = frame_loss + cfg.train.loss.lambda_depth * depth_loss
                            seq_depth  += depth_loss.item()
                    else:
                        if use_depth:
                            rendered, rendered_depth = render_view(
                                recon_d, v_mats, v_K, v_w, v_h,
                                output_type=output_type, device=device, ret_depth=True,
                            )
                        else:
                            rendered = render_view(recon_d, v_mats, v_K, v_w, v_h,
                                                   output_type=output_type, device=device)
                        photo_loss = _photometric_loss(rendered, v_gt, v_mask, cfg.train.loss)
                        frame_loss = photo_loss
                        if use_depth:
                            depth_loss  = _depth_loss(rendered_depth, v_depth, v_depth_mask)
                            frame_loss  = frame_loss + cfg.train.loss.lambda_depth * depth_loss
                            seq_depth  += depth_loss.item()

                    seq_photo    += photo_loss.item()
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

            if epoch % cfg.train.log_every == 0:
                parts = [f"Epoch {epoch:05d}/{cfg.train.epochs}", f"Loss={avg:.6f}"]
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

                        if use_depth:
                            ref_render, ref_depth = render_view(
                                splats_vis, v0_mats, v0_K, v0_w, v0_h,
                                output_type="3dgs", device=device, ret_depth=True,
                            )
                        else:
                            ref_render = render_view(splats_vis, v0_mats, v0_K, v0_w, v0_h,
                                                     output_type="3dgs", device=device)

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
                        labels = ["GT", "Reference (3DGS)", f"Recon @ {epoch}"]

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
