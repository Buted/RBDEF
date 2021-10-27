import random

from typing import List, Tuple, Dict
from cached_property import cached_property
from torch import Tensor
from torch.utils.data import Dataset

from code.dataloader.ace_dataloader import ACE_Dataset


class Meta_Dataset(ACE_Dataset):
    def __getitem__(self, index: int):
        return super(Meta_Dataset, self).__getitem__(index), self.label[index]
    
    @cached_property
    def indices2labels(self):
        return {
            idx: self.label[idx] for idx in range(len(self.label))
        }


class FewShot_Dataset(Dataset):
    def __init__(self, original_dataset, sample_ids: Tensor):
        sample_ids, remap = self._shuffle(original_dataset, sample_ids)
        self.data = []
        for idx in sample_ids:
            sample, label = original_dataset[idx]
            sample = list(sample)
            sample[1] = remap[label]
            sample = tuple(sample)
            self.data.append(sample)
    
    def _shuffle(self, original_dataset, sample_ids: Tensor) -> Tuple[List[int], Dict]:
        sample_ids = sample_ids.numpy().tolist()
        random.shuffle(sample_ids)
        remap = {}
        for idx in sample_ids:
            _, label = original_dataset[idx]
            if label not in remap:
                remap[label] = len(remap)
        return sample_ids, remap

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)