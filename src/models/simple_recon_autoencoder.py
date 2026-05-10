from uco3d.dataset_utils.data_types import GaussianSplats
from torch import nn
import torch

class POC_Network(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def reconstruct_splats(self, splats):
        means = splats.means  # (N, 3)
        sh0 = splats.sh0     # (N, 3)
        scales = splats.scales  # (N, 3)
        quats = splats.quats    # (N, 4)
        opacities = splats.opacities  # (N,)

        flat = torch.cat([means, sh0, scales, quats, opacities.unsqueeze(-1)], dim=-1)

        reconstructed_flat = self.model(flat)
        means_recon = reconstructed_flat[:, 0:3]
        sh0_recon = reconstructed_flat[:, 3:6]
        scales_recon = reconstructed_flat[:, 6:9]
        quats_recon = reconstructed_flat[:, 9:13]
        opacities_recon = reconstructed_flat[:, 13:14].squeeze(-1)

        return GaussianSplats(
            means=means_recon,
            sh0=sh0_recon,
            scales=scales_recon,
            quats=quats_recon,
            opacities=opacities_recon,
            shN=splats.shN,
            fg_mask=splats.fg_mask,
        )

network = POC_Network(14)