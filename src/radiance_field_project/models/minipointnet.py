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


class MiniPointNet2DGSNetworkPP(nn.Module):
    """MiniPointNet2DGS with a dual-branch decoder.

    The encoder is identical to MiniPointNet2DGSNetwork (per-point MLP +
    global max-pool → hidden_dim). The decoder is split into two independent MLPs:
    one for geometry (means / scales / quats) and one for appearance (sh0 /
    opacity), so the model can specialise each branch without interference.
    Both output layers are zero-initialised so the module is the identity at
    step 0.
    """

    def __init__(self, splat_dim: int = 13, encoder_in_dim: int | None = None, hidden_dim: int = 1024):
        super().__init__()
        self.splat_dim = splat_dim
        enc_in = splat_dim if encoder_in_dim is None else encoder_in_dim
        self.encoder_in_dim = enc_in
        h = hidden_dim

        self.point_encoder = nn.Sequential(
            nn.Linear(enc_in, h // 16), nn.LayerNorm(h // 16), nn.ReLU(),
            nn.Linear(h // 16, h // 4), nn.LayerNorm(h // 4),  nn.ReLU(),
            nn.Linear(h // 4,  h // 2), nn.LayerNorm(h // 2),  nn.ReLU(),
            nn.Linear(h // 2,  h),
        )

        # geometry branch: means(3) + scales2d(2) + quats(4) = 9 dims
        geom_out = nn.Linear(h // 8, 9)
        nn.init.zeros_(geom_out.weight)
        nn.init.zeros_(geom_out.bias)
        self.geom_decoder = nn.Sequential(
            nn.Linear(h + splat_dim, h // 2), nn.LayerNorm(h // 2), nn.ReLU(),
            nn.Linear(h // 2, h // 4),        nn.LayerNorm(h // 4), nn.ReLU(),
            nn.Linear(h // 4, h // 8),                               nn.ReLU(),
            geom_out,
        )

        # appearance branch: sh0(3) + opacity(1) = 4 dims
        appear_out = nn.Linear(h // 8, 4)
        nn.init.zeros_(appear_out.weight)
        nn.init.zeros_(appear_out.bias)
        self.appear_decoder = nn.Sequential(
            nn.Linear(h + splat_dim, h // 2), nn.LayerNorm(h // 2), nn.ReLU(),
            nn.Linear(h // 2, h // 4),        nn.LayerNorm(h // 4), nn.ReLU(),
            nn.Linear(h // 4, h // 8),                               nn.ReLU(),
            appear_out,
        )

    def encode(self, feats_b: torch.Tensor) -> torch.Tensor:
        """feats_b: (B, N, encoder_in_dim) → (B, 1024)"""
        return self.point_encoder(feats_b).max(dim=1).values

    def reconstruct_splats(self, splats: GaussianSplats) -> GaussianSplats:
        means     = splats.means
        sh0       = splats.sh0
        scales2d  = splats.scales[:, :2]
        quats     = splats.quats
        opacities = splats.opacities

        N    = means.shape[0]
        flat = torch.cat([means, sh0, scales2d, quats, opacities.unsqueeze(-1)], dim=-1)
        enc_input   = flat[..., :self.encoder_in_dim]
        global_feat = self.encode(enc_input.unsqueeze(0)).expand(N, -1)  # (N, 1024)
        trunk = torch.cat([global_feat, flat], dim=-1)

        d_geom   = self.geom_decoder(trunk)    # (N, 9): means(0:3), scales(3:5), quats(5:9)
        d_appear = self.appear_decoder(trunk)   # (N, 4): sh0(0:3), opacity(3)

        scales_out = torch.cat([
            scales2d + 0.01 * d_geom[:, 3:5].clamp(-1, 1),
            splats.scales[:, 2:3],
        ], dim=-1)

        return GaussianSplats(
            means     = means     + d_geom[:, 0:3],
            sh0       = sh0       + d_appear[:, 0:3],
            scales    = scales_out,
            quats     = quats     + d_geom[:, 5:9],
            opacities = opacities + d_appear[:, 3],
            shN       = splats.shN,
            fg_mask   = splats.fg_mask,
        )


class MiniPointNet2DGSNetwork(nn.Module):
    def __init__(self, splat_dim: int = 13, encoder_in_dim: int | None = None, hidden_dim: int = 1024):
        super().__init__()
        self.splat_dim = splat_dim
        enc_in = splat_dim if encoder_in_dim is None else encoder_in_dim
        self.encoder_in_dim = enc_in
        h = hidden_dim

        self.point_encoder = nn.Sequential(
            nn.Linear(enc_in, h // 16), nn.LayerNorm(h // 16), nn.ReLU(),
            nn.Linear(h // 16, h // 4), nn.LayerNorm(h // 4),  nn.ReLU(),
            nn.Linear(h // 4,  h // 2), nn.LayerNorm(h // 2),  nn.ReLU(),
            nn.Linear(h // 2,  h),
        )

        out_layer = nn.Linear(h // 8, splat_dim)
        nn.init.zeros_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        self.decoder = nn.Sequential(
            nn.Linear(h + splat_dim, h // 2), nn.LayerNorm(h // 2), nn.ReLU(),
            nn.Linear(h // 2, h // 4),        nn.LayerNorm(h // 4), nn.ReLU(),
            nn.Linear(h // 4, h // 8),                               nn.ReLU(),
            out_layer,
        )

    def encode(self, feats_b: torch.Tensor) -> torch.Tensor:
        """feats_b: (B, N, encoder_in_dim) → (B, 1024)"""
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
        _dbg = means.is_cuda
        if _dbg:
            _a = torch.cuda.memory_allocated(means.device) / 1024**3
            _r = torch.cuda.memory_reserved(means.device) / 1024**3
            print(f"  [MODEL] reconstruct_splats_2dgs N={N}  alloc={_a:.2f}GB  reserv={_r:.2f}GB", flush=True)
        flat  = torch.cat([means, sh0, scales2d, quats, opacities.unsqueeze(-1)], dim=-1)
        enc_input = flat[..., :self.encoder_in_dim]
        if _dbg:
            _a = torch.cuda.memory_allocated(means.device) / 1024**3
            print(f"  [MODEL] after flat cat  alloc={_a:.2f}GB  enc_input shape={enc_input.shape}", flush=True)
        global_feat = self.encode(enc_input.unsqueeze(0)).expand(N, -1)  # (N, 1024)
        if _dbg:
            _a = torch.cuda.memory_allocated(means.device) / 1024**3
            print(f"  [MODEL] after encode  alloc={_a:.2f}GB  global_feat shape={global_feat.shape}", flush=True)

        delta = self.decoder(torch.cat([global_feat, flat], dim=-1))  # (N, splat_dim)
        if _dbg:
            _a = torch.cuda.memory_allocated(means.device) / 1024**3
            print(f"  [MODEL] after decode  alloc={_a:.2f}GB", flush=True)

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
