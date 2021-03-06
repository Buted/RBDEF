import torch

import torch.nn as nn
import torch.nn.functional as F

from code.layers.gate import ScalableGate
from code.layers.module import Module


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class FairClassifier(Module):
    def __init__(self, embed_dim: int, out_dim: int, class_num: int):
        super(FairClassifier, self).__init__()
        self.gate = ScalableGate(embed_dim, out_dim)
        self.classifier = NormedLinear(out_dim, class_num)
    
    def forward(self, *args):
        h = self.gate(*args)
        return self.classifier(h)