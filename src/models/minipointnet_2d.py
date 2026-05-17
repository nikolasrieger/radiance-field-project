from torch import nn 
import torch 

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

        self.decoder = nn.Sequential(
            nn.Linear(1024 + splat_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),              nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),                                 nn.ReLU(),
            nn.Linear(128, splat_dim),
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