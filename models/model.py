from torch import nn
class LogisticRegression (nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x