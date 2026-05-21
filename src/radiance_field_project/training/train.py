"""Config-driven training entry-point.

Usage
-----
    python -m radiance_field_project.training.train \
        --config configs/3dgs_small.yaml \
        run.name=my_run train.lr=5e-4

Everything except ``--config`` is forwarded to OmegaConf as dotlist overrides.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.optim import Adam, AdamW
from uco3d import GaussianSplats

from radiance_field_project.config import parse_cli
from radiance_field_project.config.schema import (
    LossType,
    Optimizer,
    OutputType,
    WandbMode,
)
from radiance_field_project.data.datamodule import UCO3DDataModule
from radiance_field_project.models.model_factory import build_model
from radiance_field_project.utils.losses import reconstruction_loss
from radiance_field_project.utils.rendering import (
    get_camera_params,
    render_view,
    world_normals_to_camera,
)
from radiance_field_project.utils.splat_visualizer import SplatRenderVisualizer


def _to_device(splats: GaussianSplats, device) -> GaussianSplats:
    return GaussianSplats(
        means=splats.means.to(device).detach().float(),
        sh0=splats.sh0.to(device).detach().float(),
        scales=splats.scales.to(device).detach().float(),
        quats=splats.quats.to(device).detach().float(),
        opacities=splats.opacities.to(device).detach().float(),
        shN=None,
        fg_mask=None,
    )


def _build_optimizer(params, cfg):
    if cfg.optimizer == Optimizer.adamw:
        return AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == Optimizer.adam:
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def _photometric_loss(rendered, gt_image, fg_mask, loss_cfg):
    if loss_cfg.type == LossType.l2_masked:
        return reconstruction_loss(rendered, gt_image, mask=fg_mask,
                                   lambda_ssim=loss_cfg.lambda_ssim)
    # l1_masked
    binary = (fg_mask > 0.5).float()
    return ((rendered - gt_image).abs() * binary).sum() / binary.sum().clamp(min=1.0)


def _init_wandb(cfg, output_dir: Path):
    """Initialize wandb iff cfg.run.wandb_mode != disabled. Returns the run
    object (or ``None`` when disabled)."""
    if cfg.run.wandb_mode == WandbMode.disabled:
        return None
    return wandb.init(
        project=cfg.run.wandb_project,
        name=cfg.run.name,
        mode=cfg.run.wandb_mode.value,
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        dir=str(output_dir),
    )


def main():
    cfg = parse_cli()
    torch.manual_seed(cfg.run.seed)

    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    print(f"Output dir: {output_dir}  |  device: {device}")

    wandb_run = _init_wandb(cfg, output_dir)
    if wandb_run is not None:
        print(f"wandb: project={cfg.run.wandb_project} mode={cfg.run.wandb_mode.value} run={wandb_run.name}")
        config_artifact = wandb.Artifact(f"{cfg.run.name}-config", type="config")
        config_artifact.add_file(str(output_dir / "config.yaml"))
        wandb.log_artifact(config_artifact)

    output_type = cfg.model.output_type.value  # "3dgs" | "2dgs"
    use_normals = cfg.train.loss.lambda_normal > 0
    if use_normals and cfg.model.output_type != OutputType.gs2d:
        raise ValueError(
            "loss.lambda_normal > 0 requires model.output_type=gs2d "
            "(only the 2dgs rasterizer produces surface normals)"
        )
    print(f"Output type: {output_type}  |  normal supervision: {use_normals}")

    # ── data ────────────────────────────────────────────────────────────────
    # Normal supervision derives GT normals from depth, so force depth loading
    # whenever lambda_normal > 0, regardless of the config flag.
    dm = UCO3DDataModule(
        subset_lists_filename=cfg.data.subset_lists_filename,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_height=cfg.data.image_height,
        image_width=cfg.data.image_width,
        load_masks=cfg.data.load_masks,
        load_depths=cfg.data.load_depths or use_normals,
        load_gaussian_splats=cfg.data.load_gaussian_splats,
        pick_sequences=list(cfg.data.pick_sequences),
    )

    train_loader = dm.train_loader()
    print(f"Dataset size: {len(train_loader.dataset)}")

    n_views = cfg.data.num_supervision_views
    train_iter = iter(train_loader)
    frames = [next(train_iter) for _ in range(n_views)]

    input_splats = _to_device(frames[0]["sequence_gaussian_splats"][0], device)
    view_data = [get_camera_params(f, device, ret_normals=use_normals) for f in frames]
    if use_normals:
        v0_mats, v0_K, v0_gt, _, v0_h, v0_w, _, v0_gt_normals = view_data[0]
    else:
        v0_mats, v0_K, v0_gt, _, v0_h, v0_w = view_data[0]
        v0_gt_normals = None

    # ── model & optimizer ───────────────────────────────────────────────────
    model = build_model(cfg.model).to(device)
    optimizer = _build_optimizer(model.parameters(), cfg.train)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}  |  params: {n_params:,}")

    # Reference render of the input splats (for diagnostics). The reference
    # RGB is always rendered as 3DGS — even when the model predicts 2DGS — so
    # the panel shows the true volumetric GT rather than the flattened-surfel
    # version. When normal supervision is on, the reference normals come from
    # a separate 2DGS render of the same splats (the 3DGS rasterizer doesn't
    # emit surface normals).
    with torch.no_grad():
        ref_render = render_view(
            input_splats, v0_mats, v0_K, v0_w, v0_h,
            output_type="3dgs", device=device,
        )
        if use_normals:
            _, ref_normals = render_view(
                input_splats, v0_mats, v0_K, v0_w, v0_h,
                output_type="2dgs", device=device, ret_normals=True,
            )
            ref_normals = world_normals_to_camera(ref_normals, v0_mats)
        else:
            ref_normals = None

    # ── training loop ───────────────────────────────────────────────────────
    try:
        for epoch in range(cfg.train.epochs):
            model.train()
            total_loss = 0.0
            total_photo = 0.0
            total_normal = 0.0

            for views in view_data:
                if use_normals:
                    v_mats, v_K, v_gt, v_mask, v_h, v_w, _, v_gt_normals = views
                else:
                    v_mats, v_K, v_gt, v_mask, v_h, v_w = views

                optimizer.zero_grad(set_to_none=True)
                recon = model.reconstruct_splats(input_splats)
                if use_normals:
                    rendered, rendered_normals = render_view(
                        recon, v_mats, v_K, v_w, v_h,
                        output_type=output_type, device=device, ret_normals=True,
                    )
                else:
                    rendered = render_view(
                        recon, v_mats, v_K, v_w, v_h,
                        output_type=output_type, device=device,
                    )

                photo_loss = _photometric_loss(rendered, v_gt, v_mask, cfg.train.loss)
                if use_normals:
                    # The rasterizer emits world-frame normals; depth-derived
                    # GT normals are camera-frame. Rotate prediction into the
                    # same frame before computing the L1.
                    rendered_normals_cam = world_normals_to_camera(
                        rendered_normals, v_mats,
                    )
                    # Mask normal loss to foreground pixels that have a valid
                    # GT normal (depth-derived normals are zero where depth is
                    # invalid).
                    depth_valid = (v_gt_normals.norm(dim=1, keepdim=True) > 0.5).float()
                    normal_mask = (v_mask > 0.5).float() * depth_valid
                    normal_loss = F.l1_loss(
                        rendered_normals_cam * normal_mask,
                        v_gt_normals * normal_mask,
                    )
                    loss = photo_loss + cfg.train.loss.lambda_normal * normal_loss
                    total_normal += normal_loss.item()
                else:
                    loss = photo_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=cfg.train.grad_clip)
                optimizer.step()
                total_loss += loss.item()
                total_photo += photo_loss.item()

            n_views_seen = max(len(view_data), 1)
            avg = total_loss / n_views_seen
            avg_photo = total_photo / n_views_seen
            avg_normal = total_normal / n_views_seen
            wandb_log: dict = {
                "train/loss": avg,
                "train/photo_loss": avg_photo,
            }
            if use_normals:
                wandb_log["train/normal_loss"] = avg_normal

            if epoch % cfg.train.log_every == 0:
                if use_normals:
                    msg = (
                        f"Epoch {epoch:05d}/{cfg.train.epochs} | "
                        f"Loss = {avg:.6f} | "
                        f"Photo = {avg_photo:.6f} | "
                        f"Normal = {avg_normal:.6f}"
                    )
                else:
                    msg = f"Epoch {epoch:05d}/{cfg.train.epochs} | Loss = {avg:.6f}"
                print(msg)

                model.eval()
                with torch.no_grad():
                    eval_recon = model.reconstruct_splats(input_splats)
                    if use_normals:
                        eval_rendered, eval_normals = render_view(
                            eval_recon, v0_mats, v0_K, v0_w, v0_h,
                            output_type=output_type, device=device,
                            ret_normals=True,
                        )
                        eval_normals = world_normals_to_camera(eval_normals, v0_mats)
                    else:
                        eval_rendered = render_view(
                            eval_recon, v0_mats, v0_K, v0_w, v0_h,
                            output_type=output_type, device=device,
                        )

                    image_path = output_dir / f"epoch_{epoch:05d}.png"
                    if use_normals:
                        # Visualize normals as RGB: map [-1, 1] -> [0, 1].
                        def _nrm_vis(n):
                            return (n * 0.5 + 0.5).clamp(0, 1)

                        top = torch.cat([
                            v0_gt.permute(0, 2, 3, 1),
                            ref_render.permute(0, 2, 3, 1),
                            eval_rendered.permute(0, 2, 3, 1),
                        ], dim=2)
                        bot = torch.cat([
                            _nrm_vis(v0_gt_normals).permute(0, 2, 3, 1),
                            _nrm_vis(ref_normals).permute(0, 2, 3, 1),
                            _nrm_vis(eval_normals).permute(0, 2, 3, 1),
                        ], dim=2)
                        grid = torch.cat([top, bot], dim=1)
                        SplatRenderVisualizer.save(
                            str(image_path),
                            render_colors=grid,
                            render_alphas=torch.ones(1, v0_h * 2, v0_w * 3, 1, device=device),
                            labels=[
                                "GT image",   "Reference (3DGS)",      f"Recon RGB @ {epoch}",
                                "GT normals", "Reference normals",     f"Recon normals @ {epoch}",
                            ],
                            columns=3,
                        )
                    else:
                        SplatRenderVisualizer.save(
                            str(image_path),
                            render_colors=torch.cat([
                                v0_gt.permute(0, 2, 3, 1),
                                ref_render.permute(0, 2, 3, 1),
                                eval_rendered.permute(0, 2, 3, 1),
                            ], dim=2),
                            render_alphas=torch.ones(1, v0_h, v0_w * 3, 1, device=device),
                            labels=["GT", "Reference (3DGS)", f"Recon @ epoch {epoch}"],
                        )
                    if wandb_run is not None:
                        wandb_log["recon"] = wandb.Image(str(image_path))

            if wandb_run is not None:
                wandb.log(wandb_log, step=epoch)

            if cfg.train.ckpt_every > 0 and epoch > 0 and epoch % cfg.train.ckpt_every == 0:
                torch.save(model.state_dict(), output_dir / f"ckpt_{epoch:05d}.pt")

        torch.save(model.state_dict(), output_dir / "ckpt_final.pt")
        print(f"Done. Final checkpoint: {output_dir / 'ckpt_final.pt'}")
        if wandb_run is not None:
            ckpt_artifact = wandb.Artifact(f"{cfg.run.name}-ckpt-final", type="model")
            ckpt_artifact.add_file(str(output_dir / "ckpt_final.pt"))
            wandb.log_artifact(ckpt_artifact)
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
