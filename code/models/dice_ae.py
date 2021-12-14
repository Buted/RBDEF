import torch

import torch.nn as nn

from typing import Dict

from code.models.ae import AEModel
from code.config import Hyper


class DiceAEModel(AEModel):
    def __init__(self, hyper: Hyper):
        super(DiceAEModel, self).__init__(hyper)
        hyper.get_role2num()
        self.main_loss = nn.CrossEntropyLoss(weight=self._gen_class_weight(hyper.role2num, hyper.beta).to(hyper.gpu))
    
    @staticmethod
    def _gen_class_weight(role2num: Dict[int, int], beta: float=10):
        num = [role2num[r] for r in range(len(role2num))]
        num_sum = sum(num)
        weight = [num_sum / n + beta for n in num]
        return torch.tensor(weight).log10()

    def save(self):
        return
    
    def load(self):
        return