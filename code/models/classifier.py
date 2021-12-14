import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

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
        self.embed_dim = embed_dim
        self.out_dim = out_dim
    
    def forward(self, *args):
        h = self.gate(*args)
        return self.classifier(h)

    def load_from_meta(self, meta_n: int):
        meta_classifier = MetaClassifier(self.embed_dim, self.out_dim, meta_n)
        meta_classifier.load()
        self.gate = meta_classifier.gate
        self.classifier.weight = nn.Parameter(meta_classifier.classifier.weight[0].view(1, -1))
        self.classifier.bias = nn.Parameter(meta_classifier.classifier.bias[0].view(-1))

    def load_from_checkpoint(self, checkpoint: str):
        self.load_state_dict(
            torch.load(os.path.join(self.model_dir, self.__class__.__name__ + checkpoint))
        )
    
    @classmethod
    def load_group(cls, embed_dim: int, out_dim: int) -> Tuple:
        acc = cls(embed_dim, out_dim)
        acc.load_from_checkpoint('_acc')
        avg = cls(embed_dim, out_dim)
        avg.load_from_checkpoint('_avg')
        pos = cls(embed_dim, out_dim)
        pos.load_from_checkpoint('_pos')
        neg = cls(embed_dim, out_dim)
        neg.load_from_checkpoint('_neg')
        return torch.nn.ModuleList([acc, avg, pos, neg])


class MetaClassifier(Module):
    def __init__(self, embed_dim: int, out_dim: int, n_class: int):
        super(MetaClassifier, self).__init__()
        self.gate = ScalableGate(embed_dim, out_dim)
        self.classifier = nn.Linear(out_dim, n_class)
        
    def forward(self, *args):
        h = self.gate(*args)
        return self.classifier(h)


class AugmentMetaClassifier(ScalableClassifier):
    def load_from_meta(self, meta_roles: List[int], embed_dim: int):
        meta_classifier = MetaClassifier(embed_dim, self.classifier.in_features, len(meta_roles)+1)
        meta_classifier.load()
        self.gate = meta_classifier.gate
        for i in range(self.classifier.out_features):
            choice_meta_idx = meta_roles.index(i) if i in meta_roles else 0
            self.classifier.weight[i] = meta_classifier.classifier.weight[choice_meta_idx]
            self.classifier.bias[i] = meta_classifier.classifier.bias[choice_meta_idx]
        self.classifier.weight = nn.Parameter(self.classifier.weight)
        self.classifier.bias = nn.Parameter(self.classifier.bias)

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


class SimpleClassifier(Module):
    def __init__(self, embed_dim: int, class_num: int):
        super(SimpleClassifier, self).__init__()
        self.classifier = nn.Linear(embed_dim*2, class_num)

    def forward(self, entity_encoding, trigger_encoding):
        h = torch.cat((entity_encoding, trigger_encoding), dim=-1)
        return self.classifier(h)


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LDAMClassifier(Module):
    def __init__(self, embed_dim: int, out_dim: int, class_num: int):
        super(LDAMClassifier, self).__init__()
        self.gate = ScalableGate(embed_dim, out_dim)
        self.classifier = nn.Linear(out_dim, class_num)
    
    def forward(self, *args):
        h = self.gate(*args)
        return self.classifier(h)