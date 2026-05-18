import math
import torch


def to_2dgs_canonical(log_scales3d: torch.Tensor, quats: torch.Tensor):
    """Reorient a 3DGS splat so its thinnest axis is the 2DGS normal.

    Per-point, picks the smallest log-scale, swaps it into position 2, and
    right-multiplies the quaternion by the rotation that performs the same
    column swap on the rotation matrix (with one sign flip to keep det=+1).
    The covariance is invariant to that flip, so ``scales`` only needs to
    be permuted.

    Args:
        log_scales3d: ``(N, 3)`` log-scales (uCO3D / gsplat convention).
        quats: ``(N, 4)`` unit quaternions in ``[w, x, y, z]`` order.

    Returns:
        log_scales2d: ``(N, 2)`` the two largest log-scales per point.
        quats_canonical: ``(N, 4)`` quats with thin axis aligned to column 2.
    """
    if log_scales3d.shape[-1] != 3 or quats.shape[-1] != 4:
        raise ValueError(
            f"expected log_scales3d (N, 3) and quats (N, 4); "
            f"got {tuple(log_scales3d.shape)} and {tuple(quats.shape)}")

    k = log_scales3d.argmin(dim=-1)                                   # (N,)
    N = log_scales3d.shape[0]
    rows = torch.arange(N, device=log_scales3d.device)

    # Permute scale components: swap positions k and 2.
    idx = torch.arange(3, device=log_scales3d.device).expand(N, 3).clone()
    idx[rows, k] = 2
    idx[rows, 2] = k
    scales_perm = torch.gather(log_scales3d, 1, idx)                  # (N, 3)

    # Per-row right-multiplier quaternion (Hamilton, [w, x, y, z]):
    #   k=2: identity
    #   k=1: -90° about x  → [√2/2, -√2/2, 0, 0]
    #   k=0: 180° about (x+z)/√2 → [0, √2/2, 0, √2/2]
    s = 1.0 / math.sqrt(2.0)
    qP_table = torch.tensor(
        [[0.0,   s,  0.0,  s ],
         [ s,   -s,  0.0, 0.0],
         [1.0,  0.0, 0.0, 0.0]],
        device=quats.device, dtype=quats.dtype,
    )
    qP = qP_table[k]                                                  # (N, 4)

    w1, x1, y1, z1 = quats.unbind(-1)
    w2, x2, y2, z2 = qP.unbind(-1)
    q_new = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)
    q_new = q_new / q_new.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    return scales_perm[:, :2], q_new
