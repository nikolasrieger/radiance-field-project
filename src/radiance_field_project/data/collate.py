from uco3d.dataset_utils.frame_data import UCO3DFrameData, GaussianSplats
import torch


def uco3d_collate(batch):
    """Collate a list of FrameData into a batched FrameData.

    Uses uCO3D's built-in collate, which knows how to stack tensors,
    handle variable-length point clouds, and preserve metadata.
    """
    return UCO3DFrameData.collate(batch)


def gaussians_to_features(gs):
    """Convert a single GaussianSplats object to [N, 14] tensor."""
    return torch.cat([
        gs.means,                          # [N, 3]
        gs.sh0,                            # [N, 3]
        gs.opacities.unsqueeze(-1),        # [N, 1]
        gs.scales,                         # [N, 3]
        gs.quats,                          # [N, 4]
    ], dim=-1)                             # [N, 14]


def features_to_gaussians(features):
    """Convert a [N, 14] tensor back to a GaussianSplats instance.

    Inverse of gaussians_to_features. Feature layout:
        [means(3), sh0(3), opacities(1), scales(3), quats(4)]
    """
    assert features.shape[-1] == 14, f"expected last dim 14, got {features.shape[-1]}"

    means     = features[..., 0:3].contiguous()                  # [N, 3]
    sh0       = features[..., 3:6].unsqueeze(-2).contiguous()    # [N, 1, 3]
    opacities = features[..., 6:7].squeeze(-1).contiguous()      # [N]
    scales    = features[..., 7:10].contiguous()                 # [N, 3]
    quats     = features[..., 10:14].contiguous()                # [N, 4]

    return GaussianSplats(
        means=means,
        sh0=sh0,
        opacities=opacities,
        scales=scales,
        quats=quats,
    )


def collate_splats(sample):
    """
    sample.sequence_gaussian_splats is a list of GaussianSplats (one per batch item).
    Returns:
        features: [total_N, 14]   — all Gaussians concatenated
        batch_idx: [total_N]      — which scene each Gaussian belongs to
    """
    feats_list = [gaussians_to_features(gs) for gs in sample.sequence_gaussian_splats]
    features = torch.cat(feats_list, dim=0)
    batch_idx = torch.cat([
        torch.full((f.shape[0],), i, dtype=torch.long)
        for i, f in enumerate(feats_list)
    ])
    return features, batch_idx
