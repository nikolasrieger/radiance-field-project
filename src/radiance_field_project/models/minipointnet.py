from torch import nn
import torch
from uco3d.dataset_utils.data_types import GaussianSplats


class MiniPointNet3DGSNetwork(nn.Module):
    SPLAT_DIM = 14  # means(3) + sh0(3) + scales(3) + quats(4) + opacity(1)

    def __init__(self):
        super().__init__()

        # Per-point MLP: lifts each full splat vector to a 1024-dim feature
        self.point_encoder = nn.Sequential(
            nn.Linear(self.SPLAT_DIM, 64),
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

    def encode(self, feats_b: torch.Tensor) -> torch.Tensor:
        """feats_b: (B, N, SPLAT_DIM) → (B, 1024)"""
        per_point = self.point_encoder(feats_b)       # (B, N, 1024)
        return per_point.max(dim=1).values            # (B, 1024)

    def reconstruct_splats(self, splats: GaussianSplats) -> GaussianSplats:
        means     = splats.means        # (N, 3)
        sh0       = splats.sh0          # (N, 3)
        scales    = splats.scales       # (N, 3)
        quats     = splats.quats        # (N, 4)
        opacities = splats.opacities    # (N,)

        N = means.shape[0]
        flat = torch.cat([
            means, sh0, scales, quats, opacities.unsqueeze(-1)
        ], dim=-1)                                             # (N, 14)

        global_feat = self.encode(flat.unsqueeze(0))           # (1, 1024)
        global_feat_expanded = global_feat.expand(N, -1)       # (N, 1024)

        delta = self.decoder(torch.cat([global_feat_expanded, flat], dim=-1))  # (N, 14)

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
            nn.Linear(splat_dim, 64), nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, 256),  nn.LayerNorm(256),   nn.ReLU(),
            nn.Linear(256, 512), nn.LayerNorm(512),   nn.ReLU(),
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

    def encode(self, feats_b: torch.Tensor) -> torch.Tensor:
        """feats_b: (B, N, splat_dim) → (B, 1024)"""
        return self.point_encoder(feats_b).max(dim=1).values

    def reconstruct_splats_2dgs(
        self,
        means: torch.Tensor,
        sh0: torch.Tensor,
        scales2d: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
    ) -> dict:
        N = means.shape[0]
        flat  = torch.cat([means, sh0, scales2d, quats, opacities.unsqueeze(-1)], dim=-1)
        global_feat = self.encode(flat.unsqueeze(0)).expand(N, -1)  # (N, 1024)

        delta = self.decoder(torch.cat([global_feat, flat], dim=-1))  # (N, splat_dim)

        return dict(
            means     = means     + delta[:, 0:3],
            sh0       = sh0       + delta[:, 3:6],
            scales2d  = scales2d  + 0.01 * delta[:, 6:8].clamp(-1, 1),
            quats     = quats     + delta[:, 8:12],
            opacities = opacities + delta[:, 12],
        )

    def reconstruct_splats(self, splats: GaussianSplats) -> GaussianSplats:
        """Unified interface expected by the training loop.

        The 3rd scale column is preserved as a sentinel so the returned
        GaussianSplats keeps the (N, 3) layout that render_view expects
        (it ignores the 3rd column for 2dgs rasterization).
        """
        result = self.reconstruct_splats_2dgs(
            means     = splats.means,
            sh0       = splats.sh0,
            scales2d  = splats.scales[:, :2],
            quats     = splats.quats,
            opacities = splats.opacities,
        )
        scales_padded = torch.cat([result["scales2d"], splats.scales[:, 2:3]], dim=-1)
        return GaussianSplats(
            means     = result["means"],
            sh0       = result["sh0"],
            scales    = scales_padded,
            quats     = result["quats"],
            opacities = result["opacities"],
            shN       = splats.shN,
            fg_mask   = splats.fg_mask,
        )
