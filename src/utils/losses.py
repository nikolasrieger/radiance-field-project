import torch 

def l2_loss(pred, gt, mask=None):
    squared_diff = (pred - gt) ** 2
    if mask is None:
        return squared_diff.mean()
    binary_mask = (mask > 0.5).float()                 # (1, 1, H, W)
    return (squared_diff * binary_mask).sum() / binary_mask.sum().clamp(min=1.0)

def ssim_loss(pred, gt):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    kernel_size = 11
    sigma = 1.5
    coords = torch.arange(kernel_size, dtype=torch.float32, device=pred.device)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]                       # (11, 11)
    kernel = kernel.expand(3, 1, kernel_size, kernel_size) # (3, 1, 11, 11)

    pad = kernel_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=3)
    mu2 = F.conv2d(gt,   kernel, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sig1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=3) - mu1_sq
    sig2_sq = F.conv2d(gt   * gt,   kernel, padding=pad, groups=3) - mu2_sq
    sig12   = F.conv2d(pred * gt,   kernel, padding=pad, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))
    return 1.0 - ssim_map.mean()


def reconstruction_loss(pred, gt, mask=None, lambda_ssim=0.2):
    return (1.0 - lambda_ssim) * l2_loss(pred, gt, mask) +  lambda_ssim  * ssim_loss(pred, gt)


def normal_consistency_loss(normals, mask=None):
    # ── 1. Smoothness via finite differences ──────────────────────────────────
    dx = normals[:, :, :, 1:] - normals[:, :, :, :-1]  # (1,3,H,W-1)
    dy = normals[:, :, 1:, :] - normals[:, :, :-1, :]  # (1,3,H-1,W)

    if mask is not None:
        binary = (mask > 0.5).float()
        dx = dx * binary[:, :, :, 1:]
        dy = dy * binary[:, :, 1:, :]

    smooth = dx.abs().mean() + dy.abs().mean()

    norms     = normals.norm(dim=1, keepdim=True)           # (1,1,H,W)
    unit_pen  = (norms - 1.0).abs()
    if mask is not None:
        unit_pen = unit_pen * (mask > 0.5).float()
    unit_pen  = unit_pen.mean()

    return smooth + unit_pen