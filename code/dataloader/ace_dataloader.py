import logging
import os
import torch

import numpy as np

from functools import partial, wraps
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from code.config import Hyper
from code.utils import JsonHandler


def list2np_decorator(func):
    @wraps(func)
    def list2np(*args, **kwargs):
        return np.array(func(*args, **kwargs))
    return list2np


# @log_time
@list2np_decorator
def seq_padding(batch_seq: List[List[int]], pad_id: int = 0) -> List[List[int]]:
    L = [len(seq) for seq in batch_seq]
    ML = max(L)
    return [seq + [pad_id] * (ML - len(seq)) for seq in batch_seq]



class ACE_Dataset(Dataset):
    def __init__(self, hyper: Hyper, dataset: str):
        (
            self.tokens,
            self.label,
            self.entity_start,
            self.entity_end,
            self.entity_type,
            self.event_type,
            self.trigger_start,
            self.trigger_end
        ) = ([] for _ in range(8))

        sample_num = 0
        for sample in JsonHandler.read_json(os.path.join(hyper.data_root, dataset)):
            sample_num += 1

            tokens, wid2tid = self._tokenize(hyper.tokenizer, sample["words"])

            entity = sample["entity"]
            entity_start, entity_end = wid2tid[entity["start"]], wid2tid[entity["end"]]
            entity_type = hyper.entity2id[entity["entity_type"]]
            label = hyper.role2id[entity["role"]]
            trigger = sample["trigger"]
            trigger_start, trigger_end = wid2tid[trigger["start"]], wid2tid[trigger["end"]]

            self.tokens.append(tokens)
            self.label.append(label)
            self.entity_start.append(entity_start)
            self.entity_end.append(entity_end)
            self.entity_type.append(entity_type)
            self.event_type.append(hyper.event2id[sample["event_type"]])
            self.trigger_start.append(trigger_start)
            self.trigger_end.append(trigger_end)
        
        logging.info("The number of samples: %d" % sample_num)

    @staticmethod
    def _tokenize(tokenizer, words: List[str]) -> Tuple[List[int], Dict[int, int]]:
        tokens = []
        wid2tid = {}
        for i, w in enumerate(words):
            t = tokenizer.tokenize(w)
            wid2tid[i] = len(tokens) + 1 # add 1 for [CLS]
            tokens.extend(t)
        wid2tid[len(wid2tid)] = len(tokens) + 1 # words ending boundary

        tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens = tokenizer.build_inputs_with_special_tokens(tokens)
        return tokens, wid2tid

    def __getitem__(self, index: int):
        return (
            self.tokens[index],
            self.label[index],
            self.entity_start[index],
            self.entity_end[index],
            self.entity_type[index],
            self.event_type[index],
            self.trigger_start[index],
            self.trigger_end[index]
        )

    def __len__(self) -> int:
        return len(self.label)


class Batch_reader(object):
    def __init__(self, data):        
        transposed_data = [list(d) for d in zip(*data)]

        self.tokens = torch.LongTensor(seq_padding(transposed_data[0]))

        self.label = self._to_float_tensor(transposed_data[1])
        
        self.entity_start = self._to_long_tenor(transposed_data[2])
        self.entity_end = self._to_long_tenor(transposed_data[3])
        self.entity_type = self._to_long_tenor(transposed_data[4])
        self.event_type = self._to_long_tenor(transposed_data[5])
        self.trigger_start = self._to_long_tenor(transposed_data[6])
        self.trigger_end = self._to_long_tenor(transposed_data[7])

    @staticmethod
    def _to_long_tenor(data: List) -> torch.LongTensor:
        return torch.LongTensor(np.array(data))

    @staticmethod
    def _to_float_tensor(data: List) -> torch.FloatTensor:
        return torch.FloatTensor(np.array(data))

    def pin_memory(self):
        self.tokens = self.tokens.pin_memory()
        self.label = self.label.pin_memory()
        self.entity_start = self.entity_start.pin_memory()
        self.entity_end = self.entity_end.pin_memory()
        self.entity_type = self.entity_type.pin_memory()
        self.event_type = self.event_type.pin_memory()
        self.trigger_start = self.trigger_start.pin_memory()
        self.trigger_end = self.trigger_end.pin_memory()
        return self


def collate_fn(batch) -> Batch_reader:
    return Batch_reader(batch)


ACE_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=False)