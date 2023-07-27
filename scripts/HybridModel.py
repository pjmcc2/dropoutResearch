from torch import nn
import numpy as np
from torch.nn.functional import dropout

class HybridCurrDropModel(nn.Module):
    def __init__(self, keep_p_inp=0.9,keep_p_hidden=0.75, keep_p_out=0.5, gamma=0.001):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784,2000)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(2000,2000)
        self.output_layer = nn.Linear(2000,10)
        self.p_inp = keep_p_inp
        self.p_hidden = keep_p_hidden
        self.p_out = keep_p_out
        self._gamma = gamma
        self.t = 1


    def forward(self,input):
        flattened = self.flatten(input)
        dout0 = dropout(flattened, p=self.get_prob(self.p_inp, self.t), training=self.training)
        drop_inp1 = self.relu(self.layer1(dout0))
        dout1 = dropout(drop_inp1,p=self.get_prob(self.p_hidden,self.t), training=self.training)
        drop_inp2 = self.relu(self.hidden_layer(dout1))
        dout2 = dropout(drop_inp2, p=self.get_prob(self.p_out,self.t),
                        training=self.training)
        logits = self.output_layer(dout2)
        return logits

    def get_prob(self, init_prob, t):
      return 1- ((1. - init_prob)*np.exp(-self._gamma*t)+init_prob)

    def update_probs(self, corr):
      batch_size = len(corr)
      corr_distance = 2*corr.sum() - batch_size
      self.t += 1 if corr_distance > 0 else -1
      if self.t <= 0:
        self.t = 1