import torch
from torch import nn


class BaselineDropoutModel(nn.Module):
    def __init__(self, num_in, num_out, num_layers, output_num):
        super().__init__()
        self.mlist=[nn.Linear(num_in,num_out), nn.ReLU()]
        for i in range(num_layers):
            self.mlist.append(nn.Linear(num_out,num_out))
            self.mlist.append(nn.ReLU())
        self.mlist.append(nn.Linear(num_out,output_num))
        self.flatten = nn.Flatten()
        self.net = nn.ModuleList(self.mlist)

    def forward(self, X):
        x = self.flatten(X)
        logits = self.net(x)
        return logits
