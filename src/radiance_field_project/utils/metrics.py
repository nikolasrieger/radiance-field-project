"""Non-differentiable image/geometry quality metrics for validation logging.

These are plain scalar metrics (PSNR, SSIM, LPIPS, depth error, normal error)
used to report validation quality to wandb — as opposed to `losses.py`, whose
functions are differentiable training objectives.
"""
from __future__ import annotations

import torch

from radiance_field_project.utils.losses import ssim_loss

_LPIPS_MODELS: dict[torch.device, "torch.nn.Module"] = {}


def psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB, ``pred``/``gt`` in ``[0, 1]``."""
    if mask is not None:
        binary = (mask > 0.5).float()
        mse = ((pred - gt) ** 2 * binary).sum() / (binary.sum().clamp(min=1.0) * pred.shape[1])
    else:
        mse = (pred - gt).pow(2).mean()
    return -10.0 * torch.log10(mse.clamp(min=1e-10))


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Windowed SSIM score in ``[-1, 1]`` (higher = better)."""
    return 1.0 - ssim_loss(pred, gt)


def lpips_distance(pred: torch.Tensor, gt: torch.Tensor, device: torch.device) -> torch.Tensor:
    """AlexNet-backed LPIPS perceptual distance (lower = better).

    ``pred``/``gt`` are ``(1, 3, H, W)`` in ``[0, 1]``; internally rescaled to
    the ``[-1, 1]`` range LPIPS expects. The network is lazily built once per
    device and cached, since constructing it is expensive.
    """
    if device not in _LPIPS_MODELS:
        import lpips as _lpips
        _LPIPS_MODELS[device] = _lpips.LPIPS(net="alex").to(device).eval()
    net = _LPIPS_MODELS[device]
    with torch.no_grad():
        return net(pred * 2.0 - 1.0, gt * 2.0 - 1.0).mean()


def depth_errors(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """L1/L2 absolute & relative depth error, masked to valid GT pixels."""
    n_valid = mask.sum().clamp(min=1.0)
    diff    = (pred.nan_to_num(0.0) - gt) * mask
    safe_gt = gt.clamp(min=1e-6)
    return {
        "l1_abs": diff.abs().sum() / n_valid,
        "l2_abs": diff.pow(2).sum() / n_valid,
        "l1_rel": (diff.abs() / safe_gt).sum() / n_valid,
        "l2_rel": (diff / safe_gt).pow(2).sum() / n_valid,
    }


def normal_angular_error(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean angular error in degrees between unit normal maps, over ``mask``."""
    dot      = (pred * gt).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    ang_deg  = torch.rad2deg(torch.acos(dot))
    n_valid  = mask.sum().clamp(min=1.0)
    return (ang_deg * mask).sum() / n_valid
