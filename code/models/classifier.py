import torch.nn as nn

from code.models.gate import Gate, ScalableGate
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


class HeadClassifier(Classifier):
    def __init__(self, embed_dim: int, class_num: int):
        super(HeadClassifier, self).__init__(embed_dim, class_num)


class ScaleClassifier(Module):
    def __init__(self, embed_dim: int, out_dim: int, class_num: int):
        super(ScaleClassifier, self).__init__()
        self.gate = ScalableGate(embed_dim, out_dim)
        self.classifier = nn.Linear(out_dim, class_num)
    
    def forward(self, *args):
        h = self.gate(*args)
        return self.classifier(h)


class ScaleMainClassifier(ScaleClassifier):
    pass


class ScaleHeadClassifier(ScaleClassifier):
    pass


class SelectorClassifier(Module):
    def __init__(self, embed_dim: int, out_dim: int):
        super(SelectorClassifier, self).__init__()
        self.gate = ScalableGate(embed_dim, out_dim)
        self.classifier = nn.Linear(out_dim, 1)
    
    def forward(self, *args):
        h = self.gate(*args)
        return self.classifier(h)


class MetaClassifier(Module):
    def __init__(self, embed_dim: int, out_dim: int, n_class: int):
        super(MetaClassifier, self).__init__()
        self.gate = ScalableGate(embed_dim, out_dim)
        self.classifier = nn.Linear(out_dim, n_class)
        
    def forward(self, *args):
        h = self.gate(*args)
        return self.classifier(h)
