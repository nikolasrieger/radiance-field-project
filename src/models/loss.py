import torch

def l2_loss(pred, gt, mask):
    if mask is None:
        return torch.mean((pred - gt)**2)
    binary_mask = (mask > 0.5).float()
    squared_diff = (pred - gt)**2
    return (squared_diff * binary_mask).sum() / binary_mask.sum().clamp(min=1.0)
