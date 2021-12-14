import torch

from code.layers.gate import ScalableGate
from code.layers.module import Module
from code.layers.casual_norm_classifier import Causal_Norm_Classifier


class TDEClassifier(Module):
    def __init__(self, embed_dim: int, out_dim: int, class_num: int, alpha: float, mu: float=0.9):
        super(TDEClassifier, self).__init__()
        self.gate = ScalableGate(embed_dim, out_dim)
        self.classifier = Causal_Norm_Classifier(feat_dim=out_dim, num_classes=class_num, alpha=alpha)
        self.register_buffer('x_bar', torch.zeros(out_dim))
        self.mu = mu
    
    def forward(self, *args):
        h = self.gate(*args)
        self._update(h)
        return self.classifier(h, self.x_bar)

    def _update(self, x: torch.Tensor) -> None:
        if not self.training:
            return
        x_t = torch.mean(x.detach(), dim=0, keepdim=False)
        self.x_bar = self.mu*self.x_bar + x_t

    def train_forward(self, *args):
        h = self.gate(*args)
        _, _, y = self.classifier.train_forward(h)
        return y


class NormSelectorClassifier(TDEClassifier):
    def __init__(self, embed_dim: int, out_dim: int, alpha: float, mu: float=0.9):
        super(NormSelectorClassifier, self).__init__(embed_dim, out_dim, 2, alpha, mu)