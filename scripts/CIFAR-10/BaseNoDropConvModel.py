import torch
from torch import nn


class BaseNoDropConvModel(nn.Module): # (tries to) copy from Morerio et al (CNN..1? I think? Their code doesn't match their paper)
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Drop
            nn.Conv2d(3, 96, 5, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 - 2/2 +1 = 16 (floor division)
            # drop
            nn.Conv2d(96, 128, 5, 1, padding="same"),  # 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8
            nn.Flatten(),
            # drop
            nn.Linear(8 * 8 * 128, 2048),
            nn.ReLU(),
            # drop
            nn.Linear(2048, 1024),
            nn.ReLU(),
            # drop
            nn.Linear(1024, 10)
        )

    def forward(self, X):
        logits = self.net(X)
        return logits