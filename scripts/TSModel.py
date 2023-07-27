import torch
from torch import nn
from TsetlinUnitDropout import TsetlinUnitDropout


class TSModel(nn.Module):
    def __init__(self,p1,p2,p3,a1,a2,a3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(  # three layers
            TsetlinUnitDropout(784, p1, a1, clip=False, clip_min=0.8,clip_max=0.99),
            nn.Linear(784, 2000),
            nn.ReLU(),
            TsetlinUnitDropout(2000, p2, a2, clip=False, clip_min=0.5,clip_max=0.99),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            TsetlinUnitDropout(2000, p3, a3, clip=False, clip_min=0.5,clip_max=0.99),
            nn.Linear(2000, 10)
        )

    def forward(self, X):
        x = self.flatten(X)
        logits = self.net(x)
        return logits


