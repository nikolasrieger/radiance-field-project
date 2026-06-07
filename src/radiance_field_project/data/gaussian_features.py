"""Conversions between ``GaussianSplats`` objects and flat feature tensors."""
import torch
import torch.nn.functional as F

from uco3d.dataset_utils.frame_data import GaussianSplats


def gaussians_to_features(gs: GaussianSplats) -> torch.Tensor:
    return torch.cat([
        gs.means,                       # [N, 3]
        gs.sh0,                         # [N, 3]
        gs.opacities.unsqueeze(-1),     # [N, 1]
        gs.scales,                      # [N, 3]  
        gs.quats,                       # [N, 4]
    ], dim=-1)                          # [N, 14]


def features_to_gaussians(features: torch.Tensor) -> GaussianSplats:
    assert features.shape[-1] == 14

    return GaussianSplats(
        means     = features[..., 0:3].contiguous(),
        sh0       = features[..., 3:6].contiguous(),
        opacities = features[..., 6].contiguous(),       # raw logits — render_splats applies sigmoid
        scales    = features[..., 7:10].contiguous(),    # raw log    — render_splats applies exp
        quats     = F.normalize(features[..., 10:14], dim=-1).contiguous(),
        shN       = None,
        fg_mask   = None,
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
