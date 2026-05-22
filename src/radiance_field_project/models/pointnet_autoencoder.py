from uco3d.dataset_utils.data_types import GaussianSplats
from torch import nn
import torch
import sys
sys.path.append('/path/to/Pointnet_Pointnet2_pytorch')
from radiance_field_project.models.pointnet2_cls_ssg import get_model as get_pointnet2

class PointNet2Network(nn.Module):
    def __init__(self, pointnet_ckpt: str, normal_channel: bool = False):
        super().__init__()

        # Load pretrained PointNet++ and keep only SA layers
        full_model = get_pointnet2(num_class=40, normal_channel=normal_channel)
        ckpt = torch.load(pointnet_ckpt, map_location='cpu', weights_only=False)
        full_model.load_state_dict(ckpt['model_state_dict'])

        self.sa1 = full_model.sa1   # expects ~1024 pts
        self.sa2 = full_model.sa2
        self.sa3 = full_model.sa3

        # Decoder: global_feat (1024) + per-point splat params (14) → delta (14)
        self.decoder = nn.Sequential(
            nn.Linear(1024 + 14, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 14),
        )

    def encode(self, xyz, normals=None):
        xyz_t  = xyz.permute(0, 2, 1)                      # (B, 3, N)
        norm_t = normals.permute(0, 2, 1) if normals is not None else None

        l1_xyz, l1_pts = self.sa1(xyz_t, norm_t)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)

        return l3_pts.view(xyz.shape[0], -1)               # (B, 1024)

    def encode_multiscale(self, means_b, normals=None):
        from radiance_field_project.models.pointnet2_utils import farthest_point_sample, index_points

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

        return l3_pts.view(means_b.shape[0], -1)              # (1, 1024)


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