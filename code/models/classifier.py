import torch.nn as nn

from code.models.gate import Gate
from code.models.module import Module


class Classifier(Module):
    def __init__(self, embed_dim: int, class_num: int):
        super(Classifier, self).__init__()
        self.gate = Gate(embed_dim)
        self.classifier = nn.Linear(embed_dim, class_num)
    
    def forward(self, *args):
        h = self.gate(*args)
        return self.classifier(h)


class MainClassifier(Classifier):
    def __init__(self, embed_dim: int, class_num: int):
        super(MainClassifier, self).__init__(embed_dim, class_num)


class SelectorClassifier(Classifier):
    def __init__(self, embed_dim: int):
        super(SelectorClassifier, self).__init__(embed_dim, 1)