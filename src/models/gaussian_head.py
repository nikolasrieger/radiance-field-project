import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.pointnet2_utils import PointNetSetAbstraction
from uco3d.dataset_utils.data_types import GaussianSplats


class GaussianHead(nn.Module):
    """
    Predicts 3DGS parameters as offsets from anchors, except quats which are
    predicted directly and normalized.

    Input  x : (B, N, in_dim)   per-point features
    Output   : (B, N, 14)       [means(3), sh0(3), opacity(1), log_scale(3), quat(4)]
    """

    C0 = 0.28209479177387814  # SH DC coefficient

    def __init__(self, in_dim, init_scale=0.01, init_opacity=0.1):
        super().__init__()
        self.mean_head    = nn.Linear(in_dim, 3)
        self.sh0_head     = nn.Linear(in_dim, 3)
        self.opacity_head = nn.Linear(in_dim, 1)
        self.scale_head   = nn.Linear(in_dim, 3)
        self.quat_head    = nn.Linear(in_dim, 4)

        for h in [self.mean_head, self.sh0_head,
                  self.opacity_head, self.scale_head]:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        nn.init.zeros_(self.quat_head.weight)
        self.quat_head.bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

        opacity_logit = math.log(init_opacity / (1.0 - init_opacity))
        self.register_buffer("default_opacity_logit",
                             torch.tensor([opacity_logit]))
        self.register_buffer("default_log_scale",
                             torch.tensor([math.log(init_scale)]))

    def forward(self, x,
                anchor_means=None,
                anchor_sh0=None,
                anchor_opacity_logit=None,
                anchor_log_scale=None):
        delta_means     = self.mean_head(x)
        delta_sh0       = self.sh0_head(x)
        delta_opacity   = self.opacity_head(x)
        delta_log_scale = self.scale_head(x)

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

        quats = F.normalize(self.quat_head(x), dim=-1, eps=1e-6)

        return torch.cat([
            means,
            sh0,
            opacity_logit.clamp(-10.0, 10.0),
            log_scale.clamp(-10.0, 2.0),
            quats,
        ], dim=-1)


class get_model(nn.Module):
    """
    3DGS refiner. PointNet++ SA stack produces a (B, 1024) global feature;
    the per-point trunk consumes (global_feat ⊕ input splat params) and the
    split heads emit deltas added to anchors taken from the input splats.
    Quats are predicted directly and normalized.

    forward inputs:
        points : (B, C, P)   conditioning cloud for the encoder; C ∈ {3, 6}
        splats : (B, N, 14)  current Gaussian params (means, sh0,
                             opacity_logit, log_scale, quat) — also the anchor
                             source.
    Output:
        (B, N, 14) refined Gaussians, same layout as input.
    """
    def __init__(self,
                 normal_channel: bool = False,
                 init_scale: float = 0.01,
                 init_opacity: float = 0.1,
                 hidden_dim: int = 128,
                 freeze_encoder: bool = False):
        super().__init__()

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

        self.decoder = nn.Sequential(
            nn.Linear(1024 + 14, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU(),
        )

        self.gaussian_head = GaussianHead(hidden_dim,
                                          init_scale=init_scale,
                                          init_opacity=init_opacity)

    def encode(self, xyz):
        """xyz: (B, C, P) → (B, 1024) global feature."""
        B = xyz.size(0)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            pts  = xyz[:, :3, :]
        else:
            norm = None
            pts  = xyz
        l1_xyz, l1_pts = self.sa1(pts, norm)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
        return l3_pts.view(B, -1)  # (B, 1024)

    def forward(self, points, splats):
        B, N, D = splats.shape
        assert D == 14, f"expected 14-dim splats, got {D}"

        global_feat = self.encode(points)               # (B, 1024)
        g = global_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, 1024)
        x = torch.cat([g, splats], dim=-1)              # (B, N, 1038)
        feats = self.trunk(x)                           # (B, N, hidden)

        return self.gaussian_head(
            feats,
            anchor_means=splats[..., 0:3],
            anchor_sh0=splats[..., 3:6],
            anchor_opacity_logit=splats[..., 6:7],
            anchor_log_scale=splats[..., 7:10],
        )
    
    def reconstruct_splats(self, splats, normals=None):
        means     = splats.means        # (N, 3)
        sh0       = splats.sh0
        scales    = splats.scales
        quats     = splats.quats
        opacities = splats.opacities    # (N,)

        means_b = means.unsqueeze(0)    # (1, N, 3)

        global_feat = self.encode_multiscale(means_b, normals)  # (1, 1024)

        N = means.shape[0]
        global_feat_expanded = global_feat.expand(N, -1)        # (N, 1024)

        flat = torch.cat([
            means, sh0, scales, quats, opacities.unsqueeze(-1)
        ], dim=-1)                                               # (N, 14)

        decoder_input = torch.cat([global_feat_expanded, flat], dim=-1)  # (N, 1038)
        delta = self.decoder(decoder_input)                               # (N, 14)

        return GaussianSplats(
            means     = means     + delta[..., 0:3],
            sh0       = sh0       + delta[..., 3:6],
            scales    = scales    + 0.01 * delta[..., 6:9].clamp(-1, 1),
            quats     = quats     + delta[..., 9:13],
            opacities = opacities + delta[..., 13],
            shN       = splats.shN,
            fg_mask   = splats.fg_mask,
        )
    
    def reconstruct_splats_2dgs(self, means, sh0, scales2d, quats, opacities):
        N           = means.shape[0]
        global_feat = self.encode_multiscale(means.unsqueeze(0)).expand(N, -1)  # (N,1024)

        scales_pad = torch.zeros(N, 1, device=means.device, dtype=means.dtype)
        flat  = torch.cat([means, sh0, scales2d, scales_pad, quats,
                           opacities.unsqueeze(-1)], dim=-1)          # (N,14)
        delta = self.decoder(torch.cat([global_feat, flat], dim=-1))  # (N,hidden_dim)

        return dict(
            means     = means     + delta[:, 0:3],
            sh0       = sh0       + delta[:, 3:6],
            scales2d  = scales2d  + 0.01 * delta[:, 6:8].clamp(-1, 1),
            quats     = quats     + delta[:, 8:12],
            opacities = opacities + delta[:, 12],
        )
        
    def encode_multiscale(self, means_b, normals=None):
        from src.models.pointnet2_utils import farthest_point_sample, index_points

        # ── Normalize to unit sphere (ModelNet40 training scale) ──
        centroid = means_b.mean(dim=1, keepdim=True)          # (1, 1, 3)
        means_centered = means_b - centroid                   # (1, N, 3)
        scale = means_centered.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True).values
        means_normed = means_centered / (scale + 1e-8)        # (1, N, 3) in [-1, 1]

        # ── FPS: 50k → 1024 ──
        idx  = farthest_point_sample(means_normed, 1024)
        pts  = index_points(means_normed, idx)                # (1, 1024, 3)

        # ── SA layers expect (B, 3, N) ──
        xyz_t = pts.permute(0, 2, 1)                          # (1, 3, 1024)

        l1_xyz, l1_pts = self.sa1(xyz_t, None)                # → (1, 3, 512), (1, 128, 512)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)            # → (1, 3, 128), (1, 256, 128)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)            # → (1, 3, 1),   (1, 1024, 1)

        return l3_pts.view(means_b.shape[0], -1)     