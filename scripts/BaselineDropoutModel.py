import torch
from torch import nn


class BaselineDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(  # three layers before output
            nn.Dropout(),
            nn.Linear(784, 2000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000, 10)
        )

    def forward(self, X):
        x = self.flatten(X)
        logits = self.net(x)
        return logits
