import torch
from torch import nn
import numpy as np
from torch.nn.functional import dropout


class CurrDropConvModel(nn.Module):
    def __init__(self, keep_p_inp, keep_p_conv, keep_p_out, gamma):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.conv_inp = nn.Conv2d(3, 96, 5, 1, padding="same")
        self.conv_layer2 = nn.Conv2d(96, 128, 5, 1, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.linear_layer1 = nn.Linear(8 * 8 * 128, 2048)
        self.linear_layer2 = nn.Linear(2048, 1024)
        self.output_layer = nn.Linear(1024, 10)
        self.p_inp = keep_p_inp
        self.p_conv = keep_p_conv
        self.p_out = keep_p_out
        self.gamma = gamma


    def forward(self, X, time_step):
        dout0 = dropout(X, self.get_prob(self.p_inp, time_step), training=self.training)
        convtrio1 = self.pool(self.relu(self.conv_inp(dout0)))
        dout1 = dropout(convtrio1, self.get_prob(self.p_conv, time_step), training=self.training)
        convtrio2 = self.pool(self.relu(self.conv_layer2(dout1)))
        dout2 = dropout(self.flatten(convtrio2), self.get_prob(self.p_out, time_step), training=self.training)
        linear1 = self.relu(self.linear_layer1(dout2))
        dout3 = dropout(linear1, self.get_prob(self.p_out, time_step), training=self.training)
        linear2 = self.relu(self.linear_layer2(dout3))
        dout4 = dropout(linear2, self.get_prob(self.p_out, time_step), training=self.training)
        logits = self.output_layer(dout4)
        return logits

    def get_prob(self, init_prob, t):
        return 1 - ((1. - init_prob) * np.exp(-self.gamma * t) + init_prob)
