import torch
from torch import nn


# For use on the MNIST dataset
class BaselineDropoutModel(nn.Module):
    def __init__(self, in_size=784, out_size=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(  # three layers before output
            nn.Dropout(),
            nn.Linear(in_size, 2000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000, out_size)
        )

    def forward(self, X):
        x = self.flatten(X)
        logits = self.net(x)
        return logits
