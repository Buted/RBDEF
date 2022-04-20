import json
import os

import torch.hub as pretrained

from typing import List, Dict
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
        self.divided_data_root: str

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

        # statistic
        self.statistic: str

        # metric
        self.metric: str

        # meta
        self.meta_steps: int
        self.out_dim: int
        self.n: int
        self.k: int
        self.fast_steps: int
        self.num_task: int
        self.meta_lr: float
        self.meta_roles: List[int]
        self.filter_roles: List[int]
        self.soft: float
        self.prob: List[float]
        self.gamma: float

        # CB-loss
        self.role2num: Dict[int, int]
        self.beta: float # also beta in Dice, rho in Fair

        # Fair
        self.group_size: int
        self.group2id: Dict[str, int]
    
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
        matrix = JsonHandler.read_json(os.path.join(self.data_root, 'entity_role_co_occur.json'))
        self.co_occur_matrix = {int(entity_id): occur for entity_id, occur in matrix.items()}

    def get_role2num(self):
        role2num = json.load(open(
            os.path.join(self.data_root, 'roleid2num.json'), 
            'r',
            encoding='utf-8'
        ))
        self.role2num = {int(r): num for r, num in role2num.items()}
    
    def get_group2id(self):
        self.group2id = json.load(open(
            os.path.join(self.data_root, 'group2id.json'), 
            'r',
            encoding='utf-8'
        ))
        self.group_size = len(self.group2id) + 1