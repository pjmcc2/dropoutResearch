import torch
from torch import nn
from TsetlinUnitDropout import TsetlinUnitDropout


class TSModel(nn.Module):
    def __init__(self, initprobs, discounts, intervals, clips):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(  # three layers
            TsetlinUnitDropout(784, initprobs[0], discounts[0], clip_min=intervals[0][0], clip_max=intervals[0][1], clip=clips[0]),
            nn.Linear(784, 2000),
            nn.ReLU(),
            TsetlinUnitDropout(2000, initprobs[1], discounts[1], clip_min=intervals[1][0], clip_max=intervals[1][1], clip=clips[1]),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            TsetlinUnitDropout(2000,initprobs[2], discounts[2], clip_min=intervals[2][0], clip_max=intervals[2][1], clip=clips[2]),
            nn.Linear(2000, 10)
        )

    def forward(self, X):
        x = self.flatten(X)
        logits = self.net(x)
        return logits


