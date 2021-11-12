from code.metrics.F1 import F1
from code.config import Hyper


class HeadF1(F1):
    def __init__(self, hyper: Hyper):
        super(HeadF1, self).__init__(hyper)
        self.valid_labels = list(range(hyper.role_vocab_size - len(hyper.meta_roles) - 1))
        # self.valid_labels = list(range(hyper.role_vocab_size - len(hyper.meta_roles)))
        # self.valid_labels.remove(0)