from collections import Counter, defaultdict
from typing import List, Dict
from functools import partial
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader

from code.dataloader.ace_dataloader import ACE_Dataset, Batch_reader
from code.config import Hyper
from code.dataloader.few_role_dataloader import FewRole_Dataset


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


class MetaWithOther_Dataset(FewRoleWithOther_Dataset):
    def _build_role_remap(self, select_roles: List[int]) -> Dict[int, int]:
        role_remap = defaultdict(lambda: 1)
        role_remap[0] = 0
        for i, r in enumerate(select_roles):
            role_remap[r] = i + 2
        return role_remap


class HeadRole_Dataset(FewRoleWithOther_Dataset):
    def _build_role_remap(self, select_roles: List[int]) -> Dict[int, int]:
        role_remap = {0: 0}
        for r in select_roles:
            role_remap[r] = 1
        
        j = 2
        for i in range(1, 36):
            if i not in select_roles:
                role_remap[i] = j
                j += 1
        return role_remap


class HeadWithoutRecallRole_Dataset(FewRole_Dataset):
    def _delete_unuseless_roles(self, select_roles: List[int]):
        using_sample_ids = [i for i in range(len(self.label)) if self.label[i] not in select_roles]
        self._delete_fields(using_sample_ids)
        
        self.label_set = set([self.label[i] for i in range(len(self.label))])
        self._remap_labels(select_roles)

    def _build_role_remap(self, select_roles: List[int]):
        role_remap = {}
        for label in self.label_set:
            if label not in select_roles:
                role_remap[label] = len(role_remap)
        return role_remap


class Recall_Dataset(FewRoleWithOther_Dataset):
    def _remap_labels(self, select_roles: List[int]):
        role_remap = self._build_role_remap(select_roles)
        
        self.meta_label = [role_remap[r] for r in self.label]

    def __getitem__(self, index: int):
        return (*super(Recall_Dataset, self).__getitem__(index), self.meta_label[index])

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


class BatchWithMeta_reader(Batch_reader):
    def __init__(self, data):
        super(BatchWithMeta_reader, self).__init__(data)

        transposed_data = [list(d) for d in zip(*data)]
        self.meta_label = self._to_long_tensor(transposed_data[-1])
    
    def pin_memory(self):
        super(BatchWithMeta_reader, self).pin_memory()
        self.meta_label = self.meta_label.pin_memory()
        return self


def collate_fn(batch) -> Batch_reader:
    return BatchWithMeta_reader(batch)


ACEWithMeta_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=False)