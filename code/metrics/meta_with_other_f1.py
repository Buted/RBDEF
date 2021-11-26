from code.metrics.F1 import F1
from code.config import Hyper


class MetaWithOtherF1(F1):
    def __init__(self, hyper: Hyper):
        super(MetaWithOtherF1, self).__init__(hyper)
        self.valid_labels = list(range(1, hyper.n))