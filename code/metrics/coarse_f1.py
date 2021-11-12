from code.metrics.F1 import F1
from code.config import Hyper


class CoarseF1(F1):
    def __init__(self, hyper: Hyper):
        super(CoarseF1, self).__init__(hyper)
        self.valid_labels = list(range(3))