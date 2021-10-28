import json
import os

import torch.hub as pretrained

from typing import List
from dataclasses import dataclass
from cached_property import cached_property

from code.utils import JsonHandler


@dataclass
class Hyper(object):
    def __init__(self, path: str):
        self.dataset: str
        self.data_root: str
        self.seed: int

        # preprocess, merge
        self.raw_data_root: str
        self.raw_data_list: List[str]

        # train
        self.train: str
        self.dev: str
        self.test: str
        self.gpu: int
        self.model: str
        self.optimizer: str
        self.batch_size_train: int
        self.batch_size_eval: int
        self.epoch_num: int
        self.lr: float
        self.meta_roles: List[int]

        # statistic
        self.statistic: str

        # meta
        self.meta_steps: int
        self.out_dim: int
        self.n: int
        self.k: int
        self.fast_steps: int
        self.num_task: int
        self.meta_lr: float
    
        self.__dict__ = json.load(open(path, 'r'))
    
    def vocab_init(self):
        read_vocab = lambda x: json.load(open(
            os.path.join(self.data_root, x+'.json'), 
            'r',
            encoding='utf-8'
        ))
        self.entity2id = read_vocab('entity2id')
        self.event2id = read_vocab('event2id')
        self.role2id = read_vocab('role2id')

        reverse_vocab = lambda x: {k: v for v, k in x.items()}
        self.id2entity = reverse_vocab(self.entity2id)
        self.id2event = reverse_vocab(self.event2id)
        self.id2role = reverse_vocab(self.role2id)

        self.entity_vocab_size = len(self.entity2id)
        self.event_vocab_size = len(self.event2id)
        self.role_vocab_size = len(self.role2id)
    
    @cached_property
    def tokenizer(self):
        return pretrained.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def matrix_init(self):
        matrix = JsonHandler.read_json(os.path.join(self.data_root, 'co_occur_matrix.json'))
        self.co_occur_matrix = {int(entity_id): occur  for entity_id, occur in matrix.items()}
