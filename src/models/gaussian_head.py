import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pointnet2_utils import (
    PointNetSetAbstraction,
    farthest_point_sample,
    index_points,
)
from uco3d.dataset_utils.data_types import GaussianSplats


class _GaussianHeadBase(nn.Module):
    """Anchor-residual per-point head.

    Predicts deltas for ``means``, ``sh0``, ``opacity_logit`` and
    ``log_scale`` and adds them to anchors taken from the input splat.
    Quats are predicted directly and normalized. All delta heads are
    zero-initialized so at step 0 the module is the identity wrt the
    anchors.

    Subclasses set ``SCALE_DIM`` (3 for 3DGS, 2 for 2DGS) which controls
    both the ``scale_head`` width and the layout of the output.
    """

    SCALE_DIM: int = None  # subclasses must set

    def __init__(self, in_dim: int,
                 init_scale: float = 0.01,
                 init_opacity: float = 0.1):
        super().__init__()
        if self.SCALE_DIM is None:
            raise NotImplementedError("subclass must set SCALE_DIM")

        self.mean_head    = nn.Linear(in_dim, 3)
        self.sh0_head     = nn.Linear(in_dim, 3)
        self.opacity_head = nn.Linear(in_dim, 1)
        self.scale_head   = nn.Linear(in_dim, self.SCALE_DIM)
        self.quat_head    = nn.Linear(in_dim, 4)

        for h in [self.mean_head, self.sh0_head, self.opacity_head,
                  self.scale_head, self.quat_head]:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        opacity_logit = math.log(init_opacity / (1.0 - init_opacity))
        self.register_buffer("default_opacity_logit",
                             torch.tensor([opacity_logit]))
        self.register_buffer("default_log_scale",
                             torch.full((self.SCALE_DIM,), math.log(init_scale)))
        self.register_buffer("default_quat",
                             torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def forward(self, x,
                anchor_means=None,
                anchor_sh0=None,
                anchor_opacity_logit=None,
                anchor_log_scale=None,
                anchor_quats=None):
        delta_means     = self.mean_head(x)
        delta_sh0       = self.sh0_head(x)
        delta_opacity   = self.opacity_head(x)
        delta_log_scale = self.scale_head(x)
        delta_quats     = self.quat_head(x)

        means = delta_means + (anchor_means if anchor_means is not None else 0.0)
        sh0   = delta_sh0   + (anchor_sh0   if anchor_sh0   is not None else 0.0)

        opacity_logit = delta_opacity + (
            anchor_opacity_logit if anchor_opacity_logit is not None
            else self.default_opacity_logit
        )
        log_scale = delta_log_scale + (
            anchor_log_scale if anchor_log_scale is not None
            else self.default_log_scale
        )
        quats = F.normalize(
            delta_quats + (anchor_quats if anchor_quats is not None
                           else self.default_quat),
            dim=-1, eps=1e-6,
        )

        return torch.cat([
            means,
            sh0,
            opacity_logit.clamp(-10.0, 10.0),
            log_scale.clamp(-10.0, 2.0),
            quats,
        ], dim=-1)


class GaussianHead3DGS(_GaussianHeadBase):
    """Per-point head for 3DGS. Output channels: 14
    ``[means(3), sh0(3), opacity_logit(1), log_scale(3), quat(4)]``."""
    SCALE_DIM = 3


class GaussianHead2DGS(_GaussianHeadBase):
    """Per-point head for 2DGS. Output channels: 13
    ``[means(3), sh0(3), opacity_logit(1), log_scale(2), quat(4)]``."""
    SCALE_DIM = 2


class SplatRefiner(nn.Module):
    """Gaussian-splat residual refiner.

    A PointNet++ SA stack produces a ``(1, 1024)`` global feature; a
    per-point MLP trunk consumes ``(global_feat ⊕ per-splat params)`` and
    a format-specific head emits anchor-residual updates added to the
    input splat. With the zero-initialized head this module is the
    identity at step 0.

    Args:
        output_format: ``'3dgs'`` (14 splat dims, ``GaussianHead3DGS``) or
            ``'2dgs'`` (13 splat dims, ``GaussianHead2DGS``). Selects the
            head type and the trunk input width; only the matching
            ``reconstruct_*`` method is valid for a given instance.
    """

    _FORMATS = {
        "3dgs": (14, GaussianHead3DGS),
        "2dgs": (13, GaussianHead2DGS),
    }

    def __init__(self,
                 output_format: str = "3dgs",
                 normal_channel: bool = False,
                 init_scale: float = 0.01,
                 init_opacity: float = 0.1,
                 hidden_dim: int = 128,
                 freeze_encoder: bool = False):
        super().__init__()
        if output_format not in self._FORMATS:
            raise ValueError(
                f"output_format must be one of {list(self._FORMATS)}, "
                f"got {output_format!r}")
        self.output_format = output_format
        splat_dim, HeadCls = self._FORMATS[output_format]

        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel,
            mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024], group_all=True)

        if freeze_encoder:
            for p in list(self.sa1.parameters()) \
                   + list(self.sa2.parameters()) \
                   + list(self.sa3.parameters()):
                p.requires_grad_(False)

        self.trunk = nn.Sequential(
            nn.LayerNorm(1024 + splat_dim),
            nn.Linear(1024 + splat_dim, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU(),
        )

        self.head = HeadCls(hidden_dim,
                            init_scale=init_scale,
                            init_opacity=init_opacity)

    def encode_multiscale(self, means_b):
        """``means_b``: ``(1, N, 3)`` raw splat means.

        Normalizes the cloud to the unit sphere, FPS-downsamples to 1024
        points, runs the SA stack, and returns the global feature along
        with the normalization params so callers can apply the same
        normalization to the per-point trunk input.

        Returns:
            global_feat: ``(B, 1024)``.
            centroid:    ``(B, 1, 3)``.
            scale:       ``(B, 1, 1)`` — max distance from centroid.
        """
        centroid = means_b.mean(dim=1, keepdim=True)
        means_centered = means_b - centroid
        scale = means_centered.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True).values
        means_normed = means_centered / (scale + 1e-8)

        idx = farthest_point_sample(means_normed, 1024)
        pts = index_points(means_normed, idx)

        xyz_t = pts.permute(0, 2, 1)
        l1_xyz, l1_pts = self.sa1(xyz_t, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _,      l3_pts = self.sa3(l2_xyz, l2_pts)
        global_feat = l3_pts.view(means_b.shape[0], -1)
        return global_feat, centroid, scale

    def reconstruct_splats(self, splats: GaussianSplats) -> GaussianSplats:
        """3DGS path. Returns a refined ``GaussianSplats``."""
        if self.output_format != "3dgs":
            raise RuntimeError(
                f"reconstruct_splats requires output_format='3dgs', "
                f"this refiner is '{self.output_format}'")

        means     = splats.means
        sh0       = splats.sh0
        scales    = splats.scales
        quats     = splats.quats
        opacities = splats.opacities

        N = means.shape[0]
        global_feat, centroid, cloud_scale = self.encode_multiscale(means.unsqueeze(0))
        global_feat = global_feat.expand(N, -1)
        means_normed = (means - centroid[0, 0]) / (cloud_scale[0, 0] + 1e-8)

        flat = torch.cat([
            means_normed, sh0, scales, quats, opacities.unsqueeze(-1)
        ], dim=-1)                                                # (N, 14)
        feats = self.trunk(torch.cat([global_feat, flat], dim=-1))

        out = self.head(
            feats,
            anchor_means=means,
            anchor_sh0=sh0,
            anchor_opacity_logit=opacities.unsqueeze(-1),
            anchor_log_scale=scales,
            anchor_quats=quats,
        )                                                         # (N, 14)

        return GaussianSplats(
            means     = out[..., 0:3],
            sh0       = out[..., 3:6],
            opacities = out[..., 6],
            scales    = out[..., 7:10],
            quats     = out[..., 10:14],
            shN       = splats.shN,
            fg_mask   = splats.fg_mask,
        )

    def reconstruct_splats_2dgs(self, means, sh0, scales2d, quats, opacities):
        """2DGS path. Returns a dict with ``means``, ``sh0``, ``scales2d``,
        ``quats``, ``opacities`` (all per-point, gradient-tracked)."""
        if self.output_format != "2dgs":
            raise RuntimeError(
                f"reconstruct_splats_2dgs requires output_format='2dgs', "
                f"this refiner is '{self.output_format}'")

        N = means.shape[0]
        global_feat, centroid, cloud_scale = self.encode_multiscale(means.unsqueeze(0))
        global_feat = global_feat.expand(N, -1)
        means_normed = (means - centroid[0, 0]) / (cloud_scale[0, 0] + 1e-8)

        flat = torch.cat([
            means_normed, sh0, scales2d, quats, opacities.unsqueeze(-1)
        ], dim=-1)                                                # (N, 13)
        feats = self.trunk(torch.cat([global_feat, flat], dim=-1))

        out = self.head(
            feats,
            anchor_means=means,
            anchor_sh0=sh0,
            anchor_opacity_logit=opacities.unsqueeze(-1),
            anchor_log_scale=scales2d,
            anchor_quats=quats,
        )                                                         # (N, 13)

        return dict(
            means     = out[..., 0:3],
            sh0       = out[..., 3:6],
            opacities = out[..., 6],
            scales2d  = out[..., 7:9],
            quats     = out[..., 9:13],
        )
