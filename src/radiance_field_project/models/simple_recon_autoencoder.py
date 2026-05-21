from uco3d.dataset_utils.data_types import GaussianSplats
from torch import nn
import torch

class ReconAutoencoder(nn.Module):
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
        #nn.init.zeros_(self.model[-1].weight)
        #nn.init.zeros_(self.model[-1].bias)


    def forward(self, x):
        x = self.model(x)
        return x

    def reconstruct_splats(self, splats):
        means     = splats.means
        sh0       = splats.sh0
        scales    = splats.scales
        quats     = splats.quats
        opacities = splats.opacities

        flat  = torch.cat([means, sh0, scales, quats, opacities.unsqueeze(-1)], dim=-1)
        delta = self.model(flat)

        return GaussianSplats(
            means     = means + delta[:, 0:3],
            sh0       = sh0 + delta[:, 3:6], 
            scales    = scales + 0.01 * delta[:, 6:9].clamp(-1, 1),
            quats     = quats + delta[:, 6:10], 
            opacities = opacities + delta[:, 13], 
            shN       = splats.shN,
            fg_mask   = splats.fg_mask,
        )

class AbsoluteReconAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim=14):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

    def reconstruct_splats(self, splats):
        means     = splats.means
        sh0       = splats.sh0
        scales    = splats.scales
        quats     = splats.quats
        opacities = splats.opacities

        flat = torch.cat([means, sh0, scales, quats, opacities.unsqueeze(-1)], dim=-1)
        
        out = self.model(flat)

        return GaussianSplats(
            means     = out[:, 0:3],           # Absolute (x, y, z)
            sh0       = out[:, 3:6],           # Absolute RGB/SH0
            scales    = out[:, 6:9],           # Absolute Scales
            quats     = out[:, 9:13],          # Absolute Quaternions
            opacities = out[:, 13],            # Absolute Opacity logits
            shN       = splats.shN,            # Pass-through
            fg_mask   = splats.fg_mask,
        )