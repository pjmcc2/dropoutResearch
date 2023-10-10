import torch
from torch import nn



class BaselineNoDropoutModel(nn.Module):
    def __init__(self, in_size=784, out_size=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(in_size, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, out_size)
        )

    def forward(self, X):
        x = self.flatten(X)
        logits = self.net(x)
        return logits


