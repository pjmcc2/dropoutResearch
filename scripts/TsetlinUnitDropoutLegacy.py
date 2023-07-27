import torch
from torch import nn
import numpy as np

#TODO MAKE SURE YOU CANT GET BEYOND 1 PROB

class TsetlinUnitDropout(nn.Module):  # Drops units rather than weights
    def __init__(self, in_size, init_prob, max_step):
        super().__init__()
        assert 0 <= init_prob < 1
        self.max_step = max_step
        self.probabilities = torch.nn.Parameter(torch.tensor(np.full((1, in_size), init_prob, dtype="float32")),
                                                requires_grad=False)  # Should these be no grad parameters?
        self.not_dropped = []  # this one too? Also is there something better than empty list?

    def forward(self, inp):
        if inp.shape != self.probabilities.shape:  # This makes sure the probabilities are copied to match batch size
            self.probabilities = torch.nn.Parameter(torch.tensor(np.tile(self.probabilities[0], (inp.shape[0], 1))),
                                                    requires_grad=False)
            assert inp.shape == self.probabilities.shape
        mask = (torch.rand(inp.shape[1]) < self.probabilities).float()  # compares rand nums to probability of being
        # included
        # Assign not_dropped indices of prob tensor whose unit WAS included in network (These will be updated at step)
        self.not_dropped = [batch_el.nonzero(as_tuple=True) for batch_el in mask]  # To be indexed with
        # TODO consider flattening into 1d and then get indices somehow?
        return mask * inp / self.probabilities  # inverted dropout scaling

    def update_probs(self, correct_list):  # This implementation penalizes (increases dropout prob) if correct
        for i in range(len(correct_list)):
            if correct_list[i]:  # decrease chance of inclusion (increase dropout prob) if correct
                self.probabilities[i][self.not_dropped[i]] -= self.probabilities[i][self.not_dropped[i]] * self.max_step / torch.exp(1 - self.probabilities[i][self.not_dropped[i]])
            else:  # this number is negative (-= - = +) ie. increase chance of inclusion (decrease dropout prob)
                self.probabilities[i][self.not_dropped[i]] -= self.max_step * (self.probabilities[i][self.not_dropped[i]] - 1) / torch.exp(self.probabilities[i][self.not_dropped[i]])