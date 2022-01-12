import os
import random
import torch

import numpy as np

from typing import List, Tuple, Dict
from cached_property import cached_property
from torch import Tensor
from torch.utils.data import Dataset
from collections import defaultdict
from functools import partial
from torch.utils.data.dataloader import DataLoader

from code.config import Hyper
from code.dataloader.ace_dataloader import ACE_Dataset, seq_padding
from code.utils import JsonHandler


class Meta_Dataset(ACE_Dataset):
    def __getitem__(self, index: int):
        return super(Meta_Dataset, self).__getitem__(index), self.label[index]
    
    @cached_property
    def indices2labels(self):
        return {
            idx: self.label[idx] for idx in range(len(self.label))
        }

    @cached_property
    def labels2indices(self):
        labels2indices = defaultdict(list)
        for idx, label in enumerate(self.label):
            labels2indices[label].append(idx)
        return labels2indices


class BiasedSampling_Dataset(Meta_Dataset):
    def __init__(self, hyper: Hyper, dataset: str):
        super(BiasedSampling_Dataset, self).__init__(hyper, dataset)
        self.probability = self._build_probability(hyper)

    def _build_probability(self, hyper):
        sampling_copies = hyper.role_vocab_size - len(hyper.filter_roles)
        probability = [1 / sampling_copies if i in hyper.meta_roles else 1 / sampling_copies for i in range(hyper.role_vocab_size-len(hyper.filter_roles))]
        # for i in range(hyper.role_vocab_size):
        #     probability.append(2 / sampling_copies if i in hyper.meta_roles else 1 / sampling_copies)
        return probability


class AugmentedBiasedSampling_Dataset(BiasedSampling_Dataset):
    def __init__(self, hyper: Hyper, dataset: str):
        super(AugmentedBiasedSampling_Dataset, self).__init__(hyper, dataset)
        self._argument(hyper)
    
    def _argument(self, hyper: Hyper):
        for idx in hyper.meta_roles:
            if len(self.labels2indices[idx]) >= 2 * hyper.k:
                continue
            sample_num = len(self.labels2indices[idx])
            for i in range(2*hyper.k - sample_num):
                sample_idx = self.labels2indices[idx][i % sample_num]
                self._append_sample(sample_idx)
    
    def _append_sample(self, idx: int):
        append_field = lambda x, i: x.append(x[i])
        append_field(self.tokens, idx)
        append_field(self.label, idx)
        append_field(self.entity_start, idx)
        append_field(self.entity_end, idx)
        append_field(self.entity_id, idx)
        append_field(self.event_id, idx)
        append_field(self.trigger_start, idx)
        append_field(self.trigger_end, idx)
        append_field(self.entity_type, idx)
        append_field(self.event_type, idx)


class ImportantIndicator_Dataset(AugmentedBiasedSampling_Dataset):
    def __init__(self, hyper: Hyper, dataset: str):
        self.important = JsonHandler.read_json(os.path.join(hyper.data_root, 'important_indicator.json'))
        super(ImportantIndicator_Dataset, self).__init__(hyper, dataset)
    
    def _append_sample(self, idx: int):
        append_field = lambda x, i: x.append(x[i])
        append_field(self.tokens, idx)
        append_field(self.label, idx)
        append_field(self.entity_start, idx)
        append_field(self.entity_end, idx)
        append_field(self.entity_id, idx)
        append_field(self.event_id, idx)
        append_field(self.trigger_start, idx)
        append_field(self.trigger_end, idx)
        append_field(self.entity_type, idx)
        append_field(self.event_type, idx)
        append_field(self.important, idx)
    
    def __getitem__(self, index):
        return (*super(Meta_Dataset, self).__getitem__(index), self.important[index]), self.label[index]


class AugmentedDataset(Meta_Dataset):
    def __init__(self, hyper: Hyper, dataset: str, meta_roles: List[int] = []):
        super(AugmentedDataset, self).__init__(hyper, dataset)
        self._augment(meta_roles)

    def _augment(self, meta_roles: List[int], repeat_times: int=1):
        for i in range(len(self)):
            sample, label = self[i]
            if label not in meta_roles:
                continue
            self._repeat(sample, repeat_times)
    
    def _repeat(self, sample: Tuple, repeat_times: int):
        fields = (
            self.tokens,
            self.label,
            self.entity_start,
            self.entity_end,
            self.entity_id,
            self.event_id,
            self.trigger_start,
            self.trigger_end,
            self.entity_type
        )
        for sample_info, field in zip(sample, fields):
            field.extend([sample_info for _ in range(repeat_times)])


class FewShot_Dataset(Dataset):
    def __init__(self, original_dataset, sample_ids: Tensor, remap: Dict=None):
        if remap is None:
            self._build_remap(original_dataset, sample_ids)
        else:
            self.remap = remap
        self.data = []
        self.original_label = []
        for idx in sample_ids:
            sample, label = original_dataset[idx]
            sample = list(sample)
            sample[1] = self.remap[label]
            sample = tuple(sample)
            self.data.append(sample)
            self.original_label.append(label)
    
    def _build_remap(self, original_dataset, sample_ids: Tensor) -> Tuple[List[int], Dict]:
        sample_ids = sample_ids.numpy().tolist()
        random.shuffle(sample_ids)
        self.remap = {}
        for idx in sample_ids:
            _, label = original_dataset[idx]
            if label not in self.remap:
                self.remap[label] = len(self.remap)

    def __getitem__(self, index):
        return (*self.data[index], self.original_label[index])
    
    def __len__(self) -> int:
        return len(self.data)


class Batch_reader(object):
    def __init__(self, data):        
        transposed_data = [list(d) for d in zip(*data)]

        self.tokens = torch.LongTensor(seq_padding(transposed_data[0]))

        self.label = self._to_long_tensor(transposed_data[1])
        
        self.entity_start = self._to_long_tensor(transposed_data[2])
        self.entity_end = self._to_long_tensor(transposed_data[3])
        self.entity_id = torch.LongTensor(seq_padding(transposed_data[4]))
        self.event_id = torch.LongTensor(seq_padding(transposed_data[5]))
        self.trigger_start = self._to_long_tensor(transposed_data[6])
        self.trigger_end = self._to_long_tensor(transposed_data[7])
        self.entity_type = self._to_long_tensor(transposed_data[8])
        self.event_type = self._to_long_tensor(transposed_data[9])
        self.ori_label = self._to_long_tensor(transposed_data[-1])
        self.important = self._to_long_tensor(transposed_data[10])

    @staticmethod
    def _to_long_tensor(data: List) -> torch.LongTensor:
        return torch.LongTensor(np.array(data))

    @staticmethod
    def _to_float_tensor(data: List) -> torch.FloatTensor:
        return torch.FloatTensor(np.array(data))

    def pin_memory(self):
        self.tokens = self.tokens.pin_memory()
        self.label = self.label.pin_memory()
        self.entity_start = self.entity_start.pin_memory()
        self.entity_end = self.entity_end.pin_memory()
        self.entity_id = self.entity_id.pin_memory()
        self.event_id = self.event_id.pin_memory()
        self.trigger_start = self.trigger_start.pin_memory()
        self.trigger_end = self.trigger_end.pin_memory()
        self.entity_type = self.entity_type.pin_memory()
        self.event_type = self.event_type.pin_memory()
        self.ori_label = self.ori_label.pin_memory()
        return self


def collate_fn(batch) -> Batch_reader:
    return Batch_reader(batch)


Meta_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=False)