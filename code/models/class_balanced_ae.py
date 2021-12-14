import torch

import torch.nn as nn

from typing import Dict

from code.models.ae import AEModel
from code.config import Hyper


class ClassBalancedAEModel(AEModel):
    def __init__(self, hyper: Hyper):
        super(ClassBalancedAEModel, self).__init__(hyper)
        hyper.get_role2num()
        self.main_loss = nn.CrossEntropyLoss(weight=self._gen_class_weight(hyper.role2num, hyper.beta).to(hyper.gpu))
    
    @staticmethod
    def _gen_class_weight(role2num: Dict[int, int], beta: float=0.9):
        num = [role2num[r] for r in range(len(role2num))]
        weight = [(1 - beta) / (1 - beta**n) for n in num]
        return torch.tensor(weight)

    def save(self):
        return
    
    def load(self):
        return