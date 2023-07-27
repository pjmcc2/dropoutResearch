import torch
from torch import nn


class BaseDropConvModel(nn.Module): # (tries to) copy from Morerio et al (CNN..1? I think? Their code doesn't match their paper)
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(3, 96, 5, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 - 2/2 +1 = 16 (floor division)
            nn.Dropout(),
            nn.Conv2d(96, 128, 5, 1, padding="same"),  # 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(8 * 8 * 128, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, X):
        logits = self.net(X)
        return logits