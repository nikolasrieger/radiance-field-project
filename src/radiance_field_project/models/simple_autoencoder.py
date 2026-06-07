from torch import nn

class SimpleAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(14, 14)
        #nn.init.eye_(self.model.weight)
        #nn.init.zeros_(self.model.bias)

    def forward(self, x):
        out = self.model(x)
        return out