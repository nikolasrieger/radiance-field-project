import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiance_field_project.models.pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
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
                anchor_quats=None,
                feat_appear=None):
        # `x` carries geometry features (means/scale/quat); `feat_appear`, when
        # provided by a dual-branch decoder, carries appearance features
        # (sh0/opacity). Falls back to `x` for the single-feature path.
        x_app = feat_appear if feat_appear is not None else x
        delta_means     = self.mean_head(x)
        delta_sh0       = self.sh0_head(x_app)
        delta_opacity   = self.opacity_head(x_app)
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
    DIST_MAX: float = 2.6 * math.exp(1.0)   # ≈7.1; exp(2) caused tile-buffer OOM

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
                anchor_quats=None,
                feat_appear=None):
        # See `_GaussianHeadBase.forward`: `x` is geometry, `feat_appear` is
        # appearance (sh0/opacity) when a dual-branch decoder supplies it.
        x_app = feat_appear if feat_appear is not None else x
        delta_means   = self.mean_head(x)
        delta_sh0     = self.sh0_head(x_app)
        delta_opacity = self.opacity_head(x_app)
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

    A PointNet++ SA stack encodes the cloud; a per-point decoder produces
    per-point features and a format-specific head emits anchor-residual
    updates added to the input splat. With the zero-initialized head this
    module is the identity at step 0, regardless of decoder choice.

    Two decoders are available:
    - ``'trunk'`` (legacy): the SA stack collapses to a ``(1, 1024)`` global
      feature, broadcast to every point and concatenated with that point's
      raw params before a per-point MLP.
    - ``'fp'`` (DA3-style): a PointNet++ Feature Propagation cascade upsamples
      the multi-scale SA hierarchy coarse→fine back to the full input cloud,
      so each point receives genuine multi-scale local features. With
      ``dual_branch`` the decoded features split into a geometry branch
      (means/scale/quat) and an appearance branch (sh0/opacity), mirroring
      DA3's shared-reassemble / separate-fusion Dual-DPT head.

    Args:
        output_format: ``'3dgs'`` (14 splat dims), ``'2dgs'`` (13 dims) or
            ``'linprim'`` (12 dims). Selects the head type and decoder input
            width.
        decoder_type: ``'trunk'`` (default) or ``'fp'``.
        dual_branch: split the FP-decoded features into geometry/appearance
            branches. Only valid with ``decoder_type='fp'``.
    """

    _FORMATS = {
        "3dgs":    (14, GaussianHead3DGS),
        "2dgs":    (13, GaussianHead2DGS),
        "linprim": (12, GaussianHeadLinPrim),
    }

    def __init__(self,
                 output_format: str = "3dgs",
                 encoder=(128, 256, 1024),
                 decoder=(512, 256, 128),
                 decoder_type: str = "trunk",
                 dual_branch: bool = False,
                 num_tokens: int = 1024,
                 token_sampling: str = "fps",
                 global_skip: bool = False,
                 normal_channel: bool = False,
                 init_scale: float = 0.01,
                 init_opacity: float = 0.1,
                 freeze_encoder: bool = False,
                 encoder_in_channel: int | None = None):
        super().__init__()
        if output_format not in self._FORMATS:
            raise ValueError(
                f"output_format must be one of {list(self._FORMATS)}, "
                f"got {output_format!r}")
        if decoder_type not in ("trunk", "fp"):
            raise ValueError(
                f"decoder_type must be 'trunk' or 'fp', got {decoder_type!r}")
        if dual_branch and decoder_type != "fp":
            raise ValueError("dual_branch=True requires decoder_type='fp'")
        if global_skip and decoder_type != "fp":
            raise ValueError("global_skip=True requires decoder_type='fp' "
                             "(the trunk decoder already broadcasts the global feature)")
        if token_sampling not in ("fps", "random"):
            raise ValueError(
                f"token_sampling must be 'fps' or 'random', got {token_sampling!r}")
        self.output_format = output_format
        self.decoder_type = decoder_type
        self.dual_branch = dual_branch
        self.num_tokens = num_tokens
        self.token_sampling = token_sampling
        self.global_skip = global_skip
        splat_dim, HeadCls = self._FORMATS[output_format]

        encoder = list(encoder)
        decoder = list(decoder)
        if len(encoder) != 3:
            raise ValueError(
                f"encoder must have length 3 (SA1, SA2, SA_group_all); "
                f"got {len(encoder)}")
        if len(decoder) == 0:
            raise ValueError("decoder must contain at least one width")

        # The encoder consumes the full per-splat vector: xyz drives the
        # spatial grouping (FPS / ball query) while the remaining splat
        # attributes (sh0, scales, quats, opacity) are fed as per-point
        # features. `feat_dim` is everything in the splat vector except the
        # 3 mean/xyz channels.
        self.feat_dim = splat_dim - 3
        self.normal_channel = normal_channel
        # SA1 sees grouped_xyz_norm (3) concatenated with the per-point
        # features, plus optional normals.  encoder_in_channel lets callers
        # override this (e.g. to load checkpoints trained with xyz-only input).
        if encoder_in_channel is not None:
            in_channel = encoder_in_channel
            self.feat_dim = in_channel - 3 - (3 if normal_channel else 0)
        else:
            in_channel = 3 + self.feat_dim + (3 if normal_channel else 0)

        # Spatial subsampling params for each SA stage (the final stage is
        # always group_all=True and ignores these). Stage sizes scale with the
        # token budget; the defaults reproduce the original 512/128 at 1024.
        sa_spatial = [
            dict(npoint=max(num_tokens // 2, 1), radius=0.2, nsample=32),
            dict(npoint=max(num_tokens // 8, 1), radius=0.4, nsample=64),
        ]

        # Build SA stack — each stage's internal MLP expands prev→prev→out.
        sa_modules = []
        prev_out = in_channel
        for i, out_dim in enumerate(encoder):
            is_last = (i == len(encoder) - 1)
            spatial = (
                dict(npoint=None, radius=None, nsample=None, group_all=True)
                if is_last
                else {**sa_spatial[i], "group_all": False}
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

        self.splat_dim = splat_dim
        head_in = decoder[-1]

        if decoder_type == "trunk":
            # Legacy per-point trunk over (global feat ⊕ params). Pattern
            # matches the original: LayerNorm input, then Linear→LayerNorm→ReLU
            # for each decoder width, omitting the LayerNorm on the final layer.
            trunk_layers = [nn.LayerNorm(global_feat_dim + splat_dim)]
            prev = global_feat_dim + splat_dim
            for j, width in enumerate(decoder):
                trunk_layers.append(nn.Linear(prev, width))
                if j < len(decoder) - 1:
                    trunk_layers.append(nn.LayerNorm(width))
                trunk_layers.append(nn.ReLU())
                prev = width
            self.trunk = nn.Sequential(*trunk_layers)
        else:
            # DA3-style Feature Propagation decoder: upsample the multi-scale SA
            # hierarchy coarse→fine (RefineNet-style), so every point receives
            # genuine multi-scale local features. FP MLP widths come from
            # `decoder`; the final FP outputs `head_in` for the head.
            #   l3 (global) --fp3--> l2 --fp2--> l1 --fp1--> l0 (1024) --fp0--> N
            d0 = decoder[0]
            d1 = decoder[1] if len(decoder) > 1 else decoder[-1]
            d2 = decoder[2] if len(decoder) > 2 else decoder[-1]
            e0, e1, e2 = encoder  # SA1, SA2, SA3 output widths
            self.fp3 = PointNetFeaturePropagation(e1 + e2, [d0, d0])
            self.fp2 = PointNetFeaturePropagation(e0 + d0, [d1, d1])
            # fp1 skips the per-point splat features carried at the 1024 FPS
            # points (width feat_dim; 0 for xyz-only encoders). Source is fp2's
            # output (width d1).
            self.fp1 = PointNetFeaturePropagation(self.feat_dim + d1, [d2, d2])
            # fp0 upsamples to the full N input cloud; its skip is the raw
            # per-point splat vector (the DPT skip-to-input-resolution analog).
            self.fp0 = PointNetFeaturePropagation(splat_dim + d2, [d2, head_in])
            if global_skip:
                # Fuse (per-point FP feature ⊕ global feature) back to the head
                # width, giving every point direct access to the un-diluted
                # global summary alongside its multi-scale local features.
                self.global_fuse = nn.Sequential(
                    nn.Linear(head_in + global_feat_dim, head_in),
                    nn.LayerNorm(head_in),
                    nn.ReLU(),
                )
            if dual_branch:
                self.geom_branch = self._branch_mlp(head_in)
                self.appear_branch = self._branch_mlp(head_in)

        self.head = HeadCls(head_in,
                            init_scale=init_scale,
                            init_opacity=init_opacity)

    @staticmethod
    def _branch_mlp(width: int) -> nn.Module:
        """Small per-point branch head: Linear→LayerNorm→ReLU→Linear."""
        return nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def encode_multiscale(self, means_b, feats_b=None):
        """``means_b``: ``(1, N, 3)`` raw splat means; ``feats_b``:
        ``(1, N, feat_dim)`` per-point splat attributes (sh0, scales, quats,
        opacity) or ``None``.

        Normalizes the cloud to the unit sphere, FPS-downsamples to 1024
        points (gathering the matching point features), runs the SA stack,
        and returns the global feature along with the normalization params
        so callers can apply the same normalization to the per-point decoder
        input.

        Returns:
            global_feat: ``(B, 1024)``.
            centroid:    ``(B, 1, 3)``.
            scale:       ``(B, 1, 1)`` — max distance from centroid.
            levels:      dict of the per-level ``(xyz, pts)`` SA outputs in
                ``[B, C, S]`` layout (``l0`` = the 1024 FPS points, ``l3`` =
                global). Consumed by the FP decoder; ignored by the trunk.
        """
        centroid = means_b.mean(dim=1, keepdim=True)
        means_centered = means_b - centroid
        scale = means_centered.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True).values
        means_normed = means_centered / (scale + 1e-8)

        n_tok = min(self.num_tokens, means_b.shape[1])
        if self.token_sampling == "random":
            # Uniform random tokens: O(N) vs FPS's O(n_tok * N) sequential
            # loop, which makes large token budgets affordable.
            idx = torch.rand(
                means_b.shape[0], means_b.shape[1], device=means_b.device
            ).argsort(dim=-1)[:, :n_tok]
        else:
            idx = farthest_point_sample(means_normed, n_tok)
        pts = index_points(means_normed, idx)

        xyz_t = pts.permute(0, 2, 1)
        # Gather the per-point features at the same FPS indices and hand them
        # to SA1 as the point feature channels (B, feat_dim, 1024).
        feats_t = None
        if feats_b is not None:
            feats_t = index_points(feats_b, idx).permute(0, 2, 1)
        l1_xyz, l1_pts = self.sa1(xyz_t, feats_t)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
        global_feat = l3_pts.view(means_b.shape[0], -1)
        levels = {
            "l0_xyz": xyz_t,   "l0_pts": feats_t,
            "l1_xyz": l1_xyz,  "l1_pts": l1_pts,
            "l2_xyz": l2_xyz,  "l2_pts": l2_pts,
            "l3_xyz": l3_xyz,  "l3_pts": l3_pts,
        }
        return global_feat, centroid, scale, levels

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
        # Per-point splat attributes (everything except xyz) fed to the
        # encoder alongside the geometry.  feat_dim==0 means the encoder was
        # trained on xyz only (old checkpoints), so pass None in that case.
        if self.feat_dim > 0:
            encoder_feats = torch.cat([
                sh0, scales_in, quats, opacities.unsqueeze(-1)
            ], dim=-1)                                            # (N, feat_dim)
            encoder_feats_b = encoder_feats.unsqueeze(0)
        else:
            encoder_feats_b = None
        global_feat, centroid, cloud_scale, levels = self.encode_multiscale(
            means.unsqueeze(0), encoder_feats_b)
        means_normed = (means - centroid[0, 0]) / (cloud_scale[0, 0] + 1e-8)

        flat = torch.cat([
            means_normed, sh0, scales_in, quats, opacities.unsqueeze(-1)
        ], dim=-1)                                          # (N, 12|13|14)

        if self.decoder_type == "trunk":
            global_feat = global_feat.expand(N, -1)
            feats = self.trunk(torch.cat([global_feat, flat], dim=-1))
            feat_appear = None
        else:
            # FP cascade coarse→fine, then upsample to the full N input cloud.
            # All tensors are [B, C, S]; FP returns [B, D, S].
            up2 = self.fp3(levels["l2_xyz"], levels["l3_xyz"],
                           levels["l2_pts"], levels["l3_pts"])      # (1, d0, 128)
            up1 = self.fp2(levels["l1_xyz"], levels["l2_xyz"],
                           levels["l1_pts"], up2)                    # (1, d1, 512)
            up0 = self.fp1(levels["l0_xyz"], levels["l1_xyz"],
                           levels["l0_pts"], up1)                    # (1, d2, 1024)
            full_xyz  = means_normed.unsqueeze(0).permute(0, 2, 1)   # (1, 3, N)
            full_skip = flat.unsqueeze(0).permute(0, 2, 1)           # (1, splat_dim, N)
            per_point = self.fp0(full_xyz, levels["l0_xyz"],
                                 full_skip, up0)                     # (1, head_in, N)
            feats = per_point.squeeze(0).permute(1, 0)               # (N, head_in)
            if self.global_skip:
                feats = self.global_fuse(
                    torch.cat([feats, global_feat.expand(N, -1)], dim=-1))
            if self.dual_branch:
                feat_appear = self.appear_branch(feats)
                feats = self.geom_branch(feats)
            else:
                feat_appear = None

        if is_linprim:
            out = self.head(
                feats,
                anchor_means=means,
                anchor_sh0=sh0,
                anchor_opacity_logit=opacities.unsqueeze(-1),
                anchor_dist=scales_in,
                anchor_quats=quats,
                feat_appear=feat_appear,
            )                                               # (N, 12)
        else:
            out = self.head(
                feats,
                anchor_means=means,
                anchor_sh0=sh0,
                anchor_opacity_logit=opacities.unsqueeze(-1),
                anchor_log_scale=scales_in,
                anchor_quats=quats,
                feat_appear=feat_appear,
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
