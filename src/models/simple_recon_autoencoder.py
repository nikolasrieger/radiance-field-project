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
            scales    = scales,# + 0.01 * delta[:, 6:9].clamp(-1, 1),
            quats     = quats + delta[:, 6:10], 
            opacities = opacities + delta[:, 13], 
            shN       = splats.shN,
            fg_mask   = splats.fg_mask,
        )


network = POC_Network(14)