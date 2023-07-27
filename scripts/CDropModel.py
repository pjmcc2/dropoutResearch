from torch import nn
from torch.nn.functional import dropout
import numpy as np


class CurriculumDropoutModel(nn.Module):
    def __init__(self, keep_p_inp=0.9, keep_p_hidden=0.75, keep_p_out=0.5, gamma=0.001):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784, 2000)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(2000, 2000)
        self.output_layer = nn.Linear(2000, 10)
        self._p_inp = keep_p_inp
        self._p_hidden = keep_p_hidden
        self._p_out = keep_p_out
        self._gamma = gamma

    def forward(self, input, time_step):
        flattened = self.flatten(input)
        dout0 = dropout(flattened, p=self.get_prob(self._p_inp, time_step), training=self.training)
        drop_inp1 = self.relu(self.layer1(dout0))
        dout1 = dropout(drop_inp1, p=self.get_prob(self._p_hidden, time_step), training=self.training)
        drop_inp2 = self.relu(self.hidden_layer(dout1))
        dout2 = dropout(drop_inp2, p=self.get_prob(self._p_out, time_step),
                        training=self.training)
        logits = self.output_layer(dout2)
        return logits

    def get_prob(self, init_prob, t):
        return 1 - ((1. - init_prob) * np.exp(-self._gamma * t) + init_prob)
