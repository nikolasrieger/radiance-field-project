from torch import nn

class SimpleAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(14, 14),
            nn.ReLU(),
            nn.Linear(14,14),
        )

    def forward(self, x):
        out = self.model(x)
        return out