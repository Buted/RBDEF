import torch

import numpy as np

from typing import List
from functools import partial
from torch.utils.data.dataloader import DataLoader

from code.dataloader.ace_dataloader import ACE_Dataset, seq_padding
from code.config import Hyper


class Fair_Dataset(ACE_Dataset):
    def __init__(self, hyper: Hyper, dataset: str):
        super(Fair_Dataset, self).__init__(hyper, dataset)
        self._generate_groups(hyper)
    
    def _generate_groups(self, hyper: Hyper):
        self.group = []
        for i in range(len(self.label)):
            role = self.label[i]
            event = self.event_type[i]
            group = str(role) + '-' + str(event)

            self.group.append(hyper.group2id.get(group, 0))

    def __getitem__(self, index: int):
        return (*super(Fair_Dataset, self).__getitem__(index), self.group[index])
    

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
        self.group = self._to_long_tensor(transposed_data[10])

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
        self.group = self.group.pin_memory()
        return self


def collate_fn(batch) -> Batch_reader:
    return Batch_reader(batch)


Fair_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=False)