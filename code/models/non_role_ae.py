from code.models.selector import Selector
from code.config import Hyper
from code.models.classifier import NonRoleClassifier


class NonRoleFilter(Selector):
    def __init__(self, hyper: Hyper):
        super(NonRoleFilter, self).__init__(hyper)
        self.classifier = NonRoleClassifier(self.encoder.embed_dim, hyper.out_dim)
        self.to(self.gpu)
