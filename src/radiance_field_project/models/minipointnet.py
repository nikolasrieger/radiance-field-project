from torch import nn 
import torch 
from uco3d.dataset_utils.data_types import GaussianSplats

class MiniPointNet3DGSNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Per-point MLP: lifts each 3D point to a 1024-dim feature
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
        )

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

    def encode(self, means_b):
        per_point  = self.point_encoder(means_b)       # (B, N, 1024)
        global_feat = per_point.max(dim=1).values      # (B, 1024)
        return global_feat

    def reconstruct_splats(self, splats):
        means     = splats.means        # (N, 3)
        sh0       = splats.sh0          # (N, 3)
        scales    = splats.scales       # (N, 3)
        quats     = splats.quats        # (N, 4)
        opacities = splats.opacities    # (N,)

        means_b = means.unsqueeze(0)    # (1, N, 3)

        # Encode ALL N points — no subsampling needed
        global_feat = self.encode(means_b)             # (1, 1024)

        N = means.shape[0]
        global_feat_expanded = global_feat.expand(N, -1)  # (N, 1024)

        flat = torch.cat([
            means, sh0, scales, quats, opacities.unsqueeze(-1)
        ], dim=-1)                                         # (N, 14)

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

class MiniPointNet2DGSNetwork(nn.Module):
    def __init__(self, splat_dim: int = 13):
        super().__init__()
        self.splat_dim = splat_dim

        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),   nn.LayerNorm(64),   nn.ReLU(),
            nn.Linear(64, 256), nn.LayerNorm(256),  nn.ReLU(),
            nn.Linear(256, 512),nn.LayerNorm(512),  nn.ReLU(),
            nn.Linear(512, 1024),
        )

        out_layer = nn.Linear(128, splat_dim)
        nn.init.zeros_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        self.decoder = nn.Sequential(
            nn.Linear(1024 + splat_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),              nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),                                 nn.ReLU(),
            out_layer,
        )

    def encode(self, means_b: torch.Tensor) -> torch.Tensor:
        """means_b: (B,N,3) → (B,1024)"""
        return self.point_encoder(means_b).max(dim=1).values

    def reconstruct_splats_2dgs(self, means, sh0, scales2d, quats, opacities):
        N           = means.shape[0]
        global_feat = self.encode(means.unsqueeze(0)).expand(N, -1)  # (N,1024)

        flat  = torch.cat([means, sh0, scales2d, quats,
                           opacities.unsqueeze(-1)], dim=-1)          # (N,13)
        delta = self.decoder(torch.cat([global_feat, flat], dim=-1))  # (N,13)

        return dict(
            means     = means     + delta[:, 0:3],
            sh0       = sh0       + delta[:, 3:6],
            scales2d  = scales2d  + 0.01 * delta[:, 6:8].clamp(-1, 1),
            quats     = quats     + delta[:, 8:12],
            opacities = opacities + delta[:, 12],
        )