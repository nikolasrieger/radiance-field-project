def l2_loss(pred, gt, mask=None):
    squared_diff = (pred - gt) ** 2
    if mask is None:
        return squared_diff.mean()
    binary_mask = (mask > 0.5).float()                 # (1, 1, H, W)
    return (squared_diff * binary_mask).sum() / binary_mask.sum().clamp(min=1.0)