"""Non-differentiable image/geometry quality metrics for validation logging.

These are plain scalar metrics (PSNR, SSIM, LPIPS, depth error, normal error)
used to report validation quality to wandb — as opposed to `losses.py`, whose
functions are differentiable training objectives.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_LPIPS_MODELS: dict[torch.device, "torch.nn.Module"] = {}


def _binary_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    return None if mask is None else (mask > 0.5).float()


def _masked_pair(pred: torch.Tensor, gt: torch.Tensor,
                 mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    binary = _binary_mask(mask)
    if binary is None:
        return pred, gt
    return pred * binary, gt * binary


def _ssim_map(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    kernel_size = 11
    sigma = 1.5
    coords = torch.arange(kernel_size, dtype=torch.float32, device=pred.device)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]
    kernel = kernel.expand(pred.shape[1], 1, kernel_size, kernel_size)

    pad = kernel_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=pred.shape[1])
    mu2 = F.conv2d(gt,   kernel, padding=pad, groups=gt.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sig1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=pred.shape[1]) - mu1_sq
    sig2_sq = F.conv2d(gt   * gt,   kernel, padding=pad, groups=gt.shape[1]) - mu2_sq
    sig12   = F.conv2d(pred * gt,   kernel, padding=pad, groups=pred.shape[1]) - mu1_mu2

    return ((2 * mu1_mu2 + C1) * (2 * sig12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))


def psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB, ``pred``/``gt`` in ``[0, 1]``."""
    if mask is not None:
        binary = _binary_mask(mask)
        mse = ((pred - gt) ** 2 * binary).sum() / (binary.sum().clamp(min=1.0) * pred.shape[1])
    else:
        mse = (pred - gt).pow(2).mean()
    return -10.0 * torch.log10(mse.clamp(min=1e-10))


def ssim(pred: torch.Tensor, gt: torch.Tensor,
         mask: torch.Tensor | None = None) -> torch.Tensor:
    """Windowed SSIM score in ``[-1, 1]`` (higher = better)."""
    sim = _ssim_map(pred, gt)
    binary = _binary_mask(mask)
    if binary is None:
        return sim.mean()
    return (sim * binary).sum() / (binary.sum().clamp(min=1.0) * pred.shape[1])


def lpips_distance(pred: torch.Tensor, gt: torch.Tensor, device: torch.device,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
    """AlexNet-backed LPIPS perceptual distance (lower = better).

    ``pred``/``gt`` are ``(1, 3, H, W)`` in ``[0, 1]``; internally rescaled to
    the ``[-1, 1]`` range LPIPS expects. When ``mask`` is given, both images are
    composited onto the same black background before LPIPS is computed, matching
    the foreground-scoped validation PSNR/SSIM semantics.
    """
    if device not in _LPIPS_MODELS:
        import lpips as _lpips
        _LPIPS_MODELS[device] = _lpips.LPIPS(net="alex").to(device).eval()
    net = _LPIPS_MODELS[device]
    pred_m, gt_m = _masked_pair(pred, gt, mask)
    with torch.no_grad():
        return net(pred_m * 2.0 - 1.0, gt_m * 2.0 - 1.0).mean()


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
