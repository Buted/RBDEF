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


class ScalableClassifier(Module):
    def __init__(self, embed_dim: int, out_dim: int, class_num: int):
        super(ScalableClassifier, self).__init__()
        self.gate = ScalableGate(embed_dim, out_dim)
        self.classifier = nn.Linear(out_dim, class_num)
    
    def forward(self, *args):
        h = self.gate(*args)
        return self.classifier(h)


class ScaleMainClassifier(ScalableClassifier):
    pass


class ScaleHeadClassifier(ScalableClassifier):
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


class CoarseSelectorClassifier(ScalableClassifier):
    def __init__(self, embed_dim: int, out_dim: int):
        super(CoarseSelectorClassifier, self).__init__(embed_dim, out_dim, 3)
        self.embed_dim = embed_dim
        self.out_dim = out_dim
    
    def load(self, meta_n: int):
        meta_classifier = MetaClassifier(self.embed_dim, self.out_dim, meta_n)
        meta_classifier.load()
        self.gate = meta_classifier.gate
        self.classifier.weight = nn.Parameter(meta_classifier.classifier.weight[:3])
        self.classifier.bias = nn.Parameter(meta_classifier.classifier.bias[:3])


class NonRoleClassifier(SelectorClassifier):
    pass


class BranchSelectorClassifier(SelectorClassifier):
    pass