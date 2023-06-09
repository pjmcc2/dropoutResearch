import torch
from torch import nn


# TODO TRIPLE CHECK THE DOCS FOR EACH STEP AND WRITE IT OUT. I WANT...
# TODO ...TO KNOW WHAT EVERYTHING DOES FOR MY WRITE UP.

class BaselineNoDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(  # three layers before output
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, X):
        x = self.flatten(X)
        logits = self.net(x)
        return logits

    # heavily borrows from pytorch tutorial



