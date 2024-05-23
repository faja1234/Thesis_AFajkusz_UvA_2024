from torch import nn
class Logistic_Regression (nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class MLP(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.mlp(x)
        return out