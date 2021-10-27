from functools import partial
from typing import List
from torch.utils.data.dataloader import DataLoader

from code.config import Hyper
from code.dataloader.ace_dataloader import ACE_Dataset, collate_fn


class Selector_Dataset(ACE_Dataset):
    def __init__(self, hyper: Hyper, dataset: str, select_roles: List[int]):
        super(Selector_Dataset, self).__init__(hyper, dataset)
        self.label = [1 if role in select_roles else 0 for role in self.label]


Selector_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)