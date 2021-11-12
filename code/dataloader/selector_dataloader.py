from functools import partial
from typing import List
from torch.utils.data.dataloader import DataLoader

from code.config import Hyper
from code.dataloader.ace_dataloader import ACE_Dataset, collate_fn


class Selector_Dataset(ACE_Dataset):
    def __init__(self, hyper: Hyper, dataset: str, select_roles: List[int]):
        super(Selector_Dataset, self).__init__(hyper, dataset)
        self.label = [1 if role in select_roles else 0 for role in self.label]


class CoarseSelector_Dataset(ACE_Dataset):
    def __init__(self, hyper: Hyper, dataset: str, select_roles: List[int]):
        super(CoarseSelector_Dataset, self).__init__(hyper, dataset)
        self.select_roles = select_roles
        self.label = [self._relabel(role) for role in self.label]
    
    def _relabel(self, role: int):
        if role == 0:
            return 0
        elif role not in self.select_roles:
            return 1
        else:
            return 2


Selector_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)