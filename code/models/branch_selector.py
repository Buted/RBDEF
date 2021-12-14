from code.models.selector import Selector
from code.config import Hyper
from code.layers import BranchSelectorClassifier


class BranchSelector(Selector):
    def __init__(self, hyper: Hyper):
        super(BranchSelector, self).__init__(hyper)
        self.classifier = BranchSelectorClassifier(self.encoder.embed_dim, hyper.out_dim)
        self.to(self.gpu)