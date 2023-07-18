import torch
from torch import nn


class BaseConvModel(nn.Module): # Uses LeNet architecture
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, padding="same"),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(3, 3, 5, 2, 1),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(3, 3, 5, 2, 1),
            nn.MaxPool2d(3, 2),
            nn.Linear(5 * 5 * 3, 10)
        )

    def forward(self, X):
        logits = self.net(X)
        return logits
