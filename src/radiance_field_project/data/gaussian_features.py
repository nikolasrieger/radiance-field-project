"""Conversions between ``GaussianSplats`` objects and flat feature tensors."""
import torch
import torch.nn.functional as F

from uco3d.dataset_utils.frame_data import GaussianSplats


def gaussians_to_features(gs: GaussianSplats) -> torch.Tensor:
    """Convert a single GaussianSplats object to an ``[N, 14]`` tensor.

    Scales are stored in log-space so the network sees an unbounded target;
    :func:`features_to_gaussians` is the exact inverse on positive scales.
    """
    return torch.cat([
        gs.means,                                   # [N, 3]
        gs.sh0,                                     # [N, 3]
        gs.opacities.unsqueeze(-1),                 # [N, 1]
        torch.log(gs.scales.clamp_min(1e-8)),       # [N, 3]  log-space
        gs.quats,                                   # [N, 4]  unit-norm by convention
    ], dim=-1)                                      # [N, 14]


def features_to_gaussians(features: torch.Tensor) -> GaussianSplats:
    """Convert an ``[N, 14]`` tensor back to a ``GaussianSplats`` instance.

    Inverse of :func:`gaussians_to_features`. Feature layout:
        ``[means(3), sh0(3), opacities(1), log_scales(3), quats(4)]``.

    Decode-side guarantees for the rasterizer:
    - ``scales`` are exponentiated, so they are strictly positive.
    - ``quats`` are L2-normalized to unit length.
    """
    assert features.shape[-1] == 14, f"expected last dim 14, got {features.shape[-1]}"

    means     = features[..., 0:3].contiguous()                              # [N, 3]
    sh0       = features[..., 3:6].unsqueeze(-2).contiguous()                # [N, 1, 3]
    opacities = features[..., 6:7].squeeze(-1).contiguous()                  # [N]
    scales    = torch.exp(features[..., 7:10]).contiguous()                  # [N, 3]
    quats     = F.normalize(features[..., 10:14], dim=-1).contiguous()       # [N, 4]

    return GaussianSplats(
        means=means,
        sh0=sh0,
        opacities=opacities,
        scales=scales,
        quats=quats,
    )


def batch_to_features(sample) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a batched sample's per-scene splats into one feature tensor.

    ``sample["sequence_gaussian_splats"]`` is a list of ``GaussianSplats``
    (one per batch item). Returns:
        features:  ``[total_N, 14]`` — all Gaussians concatenated.
        batch_idx: ``[total_N]``    — which scene each Gaussian belongs to.

    Uses bracket-access so this works whether ``sample`` is a ``UCO3DFrameData``
    (which is a ``Mapping`` and forwards ``__getitem__`` to ``getattr``) or a
    plain ``dict`` (as some uco3d versions return after collation).
    """
    feats_list = [gaussians_to_features(gs) for gs in sample["sequence_gaussian_splats"]]
    features = torch.cat(feats_list, dim=0)
    batch_idx = torch.cat([
        torch.full((f.shape[0],), i, dtype=torch.long)
        for i, f in enumerate(feats_list)
    ])
    return features, batch_idx
