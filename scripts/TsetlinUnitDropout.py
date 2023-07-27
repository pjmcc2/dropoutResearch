import torch
from torch import nn
import numpy as np


class TsetlinUnitDropout(nn.Module):
    def __init__(self, in_size, init_prob, step_discount, clip=False, clip_min=None, clip_max=None):
        super().__init__()
        assert 0 <= init_prob < 1
        self.discount = step_discount
        self.pmin = clip_min
        self.pmax = clip_max
        self.clip = clip
        self.probabilities = torch.nn.Parameter(torch.tensor(np.full((1, in_size), init_prob, dtype="float32")),
                                                requires_grad=False)
        self.not_dropped = []

    def forward(self, inp):
        dev = self.probabilities.device
        comparator = torch.rand(inp.shape[1]).to(dev)
        mask = (comparator < self.probabilities).float().to(dev) # compares rand nums to probability of being
        # included
        # Assign not_dropped indices of prob tensor whose unit WAS included in network (These will be updated at step)
        self.not_dropped = mask.nonzero(as_tuple=True)  # To be indexed with
        return mask * inp / self.probabilities  # inverted dropout scaling

    def update_probs(self, correct_list):  # This implementation penalizes (increases dropout prob) if correct
        batch_size = len(correct_list)
        corr_distance = 2*correct_list.sum() - batch_size
        if corr_distance > 0: # decrease chance of inclusion (increase dropout prob) if correct
          self.probabilities[self.not_dropped] -=  (self.probabilities[self.not_dropped] * self.discount / torch.exp(
                1 - self.probabilities[self.not_dropped]))

        elif corr_distance < 0:  # this number is negative (-= - = +) ie. increase chance of inclusion (decrease dropout prob)
            self.probabilities[self.not_dropped] -=  (self.discount * (self.probabilities[self.not_dropped] - 1) / torch.exp(
                self.probabilities[self.not_dropped]))

        if self.clip:
          torch.clamp_(self.probabilities, self.pmin, self.pmax)