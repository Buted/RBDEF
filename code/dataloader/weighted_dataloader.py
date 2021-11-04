from collections import Counter, defaultdict
from typing import List, Dict
from functools import partial
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader

from code.dataloader.ace_dataloader import ACE_Dataset, Batch_reader
from code.config import Hyper


class FewRoleWithOther_Dataset(ACE_Dataset):
    def __init__(self, hyper: Hyper, dataset: str, select_roles: List[int]):
        super(FewRoleWithOther_Dataset, self).__init__(hyper, dataset)
        self._remap_labels(select_roles)
    
    def _remap_labels(self, select_roles: List[int]):
        role_remap = self._build_role_remap(select_roles)
        
        self.label = [role_remap[r] for r in self.label]

    def _build_role_remap(self, select_roles: List[int]) -> Dict[int, int]:
        role_remap = defaultdict(int)
        for i, r in enumerate(select_roles):
            role_remap[r] = i + 1
        return role_remap


class WeightedRoleSampler:
    def __init__(self, dataset):
        weights = self._compute_weights(dataset)
        self.sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

    def _compute_weights(self, dataset):
        labels = dataset.label
        label_cnt = Counter(labels)
        label2weight = {label: label_cnt[0] // cnt for label, cnt in label_cnt.items()}
        # print(label2weight)
        # exit(0)
        return [label2weight[label] for label in labels]