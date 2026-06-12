import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiance_field_project.models.pointnet2_utils import (
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


class GaussianHeadLinPrim(nn.Module):
    """Per-point head for LinPrim octahedra. Output channels: 12
    ``[means(3), sh0(3), opacity_logit(1), dist(1), quat(4)]``.

    A single symmetric distance parameter describes each octahedron (abs
    activation, not exp). Zero-initialized so the module is the identity
    at step 0.
    """

    # Upper bound on |dist| (world units). The 3DGS/2DGS heads cap effective
    # scale via log_scale.clamp(max=2.0) -> exp(2) sigma; with DIST_TO_SCALE=2.6
    # the equivalent dist cap is 2.6 * exp(2). Without this bound, dist can grow
    # unbounded during training and balloon the octahedra, which explodes the
    # rasterizer's binning buffer (multi-GB allocation -> CUDA OOM).
    DIST_MAX: float = 2.6 * math.exp(2.0)

    def __init__(self, in_dim: int,
                 init_scale: float = 0.01,
                 init_opacity: float = 0.1):
        super().__init__()
        self.mean_head    = nn.Linear(in_dim, 3)
        self.sh0_head     = nn.Linear(in_dim, 3)
        self.opacity_head = nn.Linear(in_dim, 1)
        self.dist_head    = nn.Linear(in_dim, 1)
        self.quat_head    = nn.Linear(in_dim, 4)

        for h in [self.mean_head, self.sh0_head, self.opacity_head,
                  self.dist_head, self.quat_head]:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        opacity_logit = math.log(init_opacity / (1.0 - init_opacity))
        self.register_buffer("default_opacity_logit",
                             torch.tensor([opacity_logit]))
        # LinPrim DIST_TO_SCALE = 2.6: octahedra cover 1/2.6 the volume of
        # a Gaussian with the same standard deviation.
        self.register_buffer("default_dist",
                             torch.tensor([init_scale * 2.6]))
        self.register_buffer("default_quat",
                             torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def forward(self, x,
                anchor_means=None,
                anchor_sh0=None,
                anchor_opacity_logit=None,
                anchor_dist=None,
                anchor_quats=None):
        delta_means   = self.mean_head(x)
        delta_sh0     = self.sh0_head(x)
        delta_opacity = self.opacity_head(x)
        delta_dist    = self.dist_head(x)
        delta_quats   = self.quat_head(x)

        means = delta_means + (anchor_means if anchor_means is not None else 0.0)
        sh0   = delta_sh0   + (anchor_sh0   if anchor_sh0   is not None else 0.0)

        opacity_logit = delta_opacity + (
            anchor_opacity_logit if anchor_opacity_logit is not None
            else self.default_opacity_logit
        )
        dist = delta_dist + (
            anchor_dist if anchor_dist is not None else self.default_dist
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
            dist.clamp(-self.DIST_MAX, self.DIST_MAX),
            quats,
        ], dim=-1)


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
        "3dgs":    (14, GaussianHead3DGS),
        "2dgs":    (13, GaussianHead2DGS),
        "linprim": (12, GaussianHeadLinPrim),
    }

    # Spatial subsampling params for each SA stage (the final stage is
    # always group_all=True and ignores these).
    _SA_SPATIAL = [
        dict(npoint=512, radius=0.2, nsample=32),
        dict(npoint=128, radius=0.4, nsample=64),
    ]

    def __init__(self,
                 output_format: str = "3dgs",
                 encoder=(128, 256, 1024),
                 decoder=(512, 256, 128),
                 normal_channel: bool = False,
                 init_scale: float = 0.01,
                 init_opacity: float = 0.1,
                 freeze_encoder: bool = False):
        super().__init__()
        if output_format not in self._FORMATS:
            raise ValueError(
                f"output_format must be one of {list(self._FORMATS)}, "
                f"got {output_format!r}")
        self.output_format = output_format
        splat_dim, HeadCls = self._FORMATS[output_format]

        encoder = list(encoder)
        decoder = list(decoder)
        if len(encoder) != 3:
            raise ValueError(
                f"encoder must have length 3 (SA1, SA2, SA_group_all); "
                f"got {len(encoder)}")
        if len(decoder) == 0:
            raise ValueError("decoder must contain at least one width")

        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # Build SA stack — each stage's internal MLP expands prev→prev→out.
        sa_modules = []
        prev_out = in_channel
        for i, out_dim in enumerate(encoder):
            is_last = (i == len(encoder) - 1)
            spatial = (
                dict(npoint=None, radius=None, nsample=None, group_all=True)
                if is_last
                else {**self._SA_SPATIAL[i], "group_all": False}
            )
            # SA input channel: raw `in_channel` for the first stage; for
            # later stages PointNet++ concatenates xyz, so prev_out + 3.
            sa_in = in_channel if i == 0 else prev_out + 3
            mid = max(out_dim // 2, 1)
            sa_modules.append(PointNetSetAbstraction(
                in_channel=sa_in,
                mlp=[mid, mid, out_dim],
                **spatial,
            ))
            prev_out = out_dim
        self.sa1, self.sa2, self.sa3 = sa_modules
        global_feat_dim = encoder[-1]

        if freeze_encoder:
            for p in list(self.sa1.parameters()) \
                   + list(self.sa2.parameters()) \
                   + list(self.sa3.parameters()):
                p.requires_grad_(False)

        # Build the per-point trunk from `decoder`. Pattern matches the
        # original: LayerNorm input, then Linear→LayerNorm→ReLU for each
        # decoder width, omitting the LayerNorm on the final layer.
        trunk_layers = [nn.LayerNorm(global_feat_dim + splat_dim)]
        prev = global_feat_dim + splat_dim
        for j, width in enumerate(decoder):
            trunk_layers.append(nn.Linear(prev, width))
            if j < len(decoder) - 1:
                trunk_layers.append(nn.LayerNorm(width))
            trunk_layers.append(nn.ReLU())
            prev = width
        self.trunk = nn.Sequential(*trunk_layers)

        self.head = HeadCls(decoder[-1],
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
        """Refine ``splats`` and return a ``GaussianSplats`` of the same layout.

        Handles ``output_format`` values ``'3dgs'``, ``'2dgs'``, and
        ``'linprim'``.

        - 3dgs: refines all 3 scale dimensions.
        - 2dgs: refines 2 scale dimensions; pads a log-sentinel in position 2
          so the returned ``scales`` tensor is always ``(N, 3)``.
        - linprim: ``splats.scales`` holds a single raw distance per point
          ``(N, 1)`` (set up by ``_cache_splat``); the head predicts an
          additive residual. ``render_view`` applies ``abs`` and expands to
          ``(N, 3)`` for the rasterizer.
        """
        is_2dgs    = self.output_format == "2dgs"
        is_linprim = self.output_format == "linprim"
        scale_dim  = 1 if is_linprim else (2 if is_2dgs else 3)

        means     = splats.means
        sh0       = splats.sh0
        quats     = splats.quats
        opacities = splats.opacities
        scales_in = splats.scales[:, :scale_dim]

        N = means.shape[0]
        global_feat, centroid, cloud_scale = self.encode_multiscale(means.unsqueeze(0))
        global_feat = global_feat.expand(N, -1)
        means_normed = (means - centroid[0, 0]) / (cloud_scale[0, 0] + 1e-8)

        flat = torch.cat([
            means_normed, sh0, scales_in, quats, opacities.unsqueeze(-1)
        ], dim=-1)                                          # (N, 12|13|14)
        feats = self.trunk(torch.cat([global_feat, flat], dim=-1))

        if is_linprim:
            out = self.head(
                feats,
                anchor_means=means,
                anchor_sh0=sh0,
                anchor_opacity_logit=opacities.unsqueeze(-1),
                anchor_dist=scales_in,
                anchor_quats=quats,
            )                                               # (N, 12)
        else:
            out = self.head(
                feats,
                anchor_means=means,
                anchor_sh0=sh0,
                anchor_opacity_logit=opacities.unsqueeze(-1),
                anchor_log_scale=scales_in,
                anchor_quats=quats,
            )                                               # (N, 13|14)

        means_r   = out[..., 0:3]
        sh0_r     = out[..., 3:6]
        opacity_r = out[..., 6]
        scales_r  = out[..., 7:7 + scale_dim]
        quats_r   = out[..., 7 + scale_dim:11 + scale_dim]

        if is_2dgs:
            z = torch.full(
                (scales_r.shape[0], 1), -10.0,
                device=scales_r.device, dtype=scales_r.dtype,
            )
            scales_r = torch.cat([scales_r, z], dim=-1)

        return GaussianSplats(
            means     = means_r,
            sh0       = sh0_r,
            opacities = opacity_r,
            scales    = scales_r,
            quats     = quats_r,
            shN       = splats.shN,
            fg_mask   = splats.fg_mask,
        )
