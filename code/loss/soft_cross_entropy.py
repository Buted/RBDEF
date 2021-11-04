import torch

import torch.nn as nn


class SoftCrossEntropyLoss():
    def __init__(self):
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def __call__(self, logits, target):
        log_likelihood = -self.log_softmax(logits)
        batch_size = logits.shape[0]
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch_size
        return loss
