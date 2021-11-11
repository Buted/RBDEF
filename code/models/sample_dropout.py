import torch

import numpy as np
import torch.nn as nn

from code.config import Hyper


class SampleDropout:
    def __init__(self, hyper: Hyper):
        self.p = hyper.prob
        self.dropout_ratio = nn.Embedding(hyper.n, 1)
        self._init_dropout()

    def _init_dropout(self):
        np_prob = np.array(self.p)
        np_prob = np_prob.reshape(-1, 1)        
        weight = torch.from_numpy(np_prob)
        self.dropout_ratio.weight.data.copy_(weight)
        self.dropout_ratio.weight.requires_grad = False
    
    def __call__(self, x, label):
        ratio, mask = self._generate_mask(x, label) 
        x *= mask
        
        amplification = torch.reciprocal(1 -  ratio)
        amplification = amplification.detach()
        x *= amplification
        return x

    def _generate_mask(self, input, label):
        ratio = self.dropout_ratio(label)

        mask = torch.rand_like(input)
        mask = mask > ratio
        mask = mask.float()
        mask.requires_grad = False
        return ratio, mask