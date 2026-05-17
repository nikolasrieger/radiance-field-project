from uco3d.dataset_utils.data_types import GaussianSplats
from torch import nn
import torch
import sys
sys.path.append('/path/to/Pointnet_Pointnet2_pytorch')  # adjust to your path

from models.pointnet2_cls_ssg import get_model as get_pointnet2

class PointNet2_POC_Network(nn.Module):
    """
    Uses PointNet++ as a per-point encoder, then decodes residual deltas
    on top of the original Gaussian splat parameters.
    """
    def __init__(self, pointnet_ckpt: str, normal_channel: bool = False):
        super().__init__()

        # --- Encoder: PointNet++ (frozen or fine-tunable) ---
        full_model = get_pointnet2(num_class=40, normal_channel=normal_channel)
        ckpt = torch.load(pointnet_ckpt, map_location='cpu', weights_only=False)
        full_model.load_state_dict(ckpt['model_state_dict'])

        # Keep only the three Set Abstraction layers
        self.sa1 = full_model.sa1
        self.sa2 = full_model.sa2
        self.sa3 = full_model.sa3
        # → produces a (B, 1024) global feature vector

        # --- Decoder: global feature → per-point deltas ---
        # global_feat (1024) + per-point raw features (14) → delta (14)
        self.decoder = nn.Sequential(
            nn.Linear(1024 + 14, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 14),   # delta for [means(3), sh0(3), scales(3), quats(4), opacity(1)]
        )

    def encode(self, xyz, normals=None):
        """
        xyz:     (B, N, 3)
        normals: (B, N, 3) optional
        returns: (B, 1024) global feature
        """
        # SA layers expect (B, C, N)
        xyz_t = xyz.permute(0, 2, 1)          # (B, 3, N)

        if normals is not None:
            norm_t = normals.permute(0, 2, 1)  # (B, 3, N)
        else:
            norm_t = None

        l1_xyz, l1_pts = self.sa1(xyz_t, norm_t)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)

        return l3_pts.view(xyz.shape[0], -1)   # (B, 1024)

    def reconstruct_splats(self, splats, normals=None):
        means     = splats.means        # (N, 3)
        sh0       = splats.sh0
        scales    = splats.scales
        quats     = splats.quats
        opacities = splats.opacities    # (N,)

        # Add batch dim for PointNet++
        means_b = means.unsqueeze(0)    # (1, N, 3)

        flat = torch.cat([
            means, sh0, scales, quats, opacities.unsqueeze(-1)
        ], dim=-1)                      # (N, 14)

        # Encode: (1, 1024)
        global_feat = self.encode(means_b, normals)
        N = means.shape[0]
        global_feat_expanded = global_feat.expand(N, -1)   # (N, 1024)

        # Decode
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