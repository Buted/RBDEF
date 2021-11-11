import os
import torch

import torch.nn as nn
import numpy as np

from typing import Dict

from code.config import Hyper
from code.utils import JsonHandler


class MaskHandler:
    def __init__(self, hyper: Hyper) -> None:
        self.hyper = hyper
        self.gpu = hyper.gpu
        self.soft = hyper.soft

        entity_co_occur_filename = os.path.join(hyper.data_root, 'entity_role_co_occur.json')
        self.entity_co_occur_embedding = self._init_embedding(entity_co_occur_filename, len(hyper.entity2id))
        event_co_occur_filename = os.path.join(hyper.data_root, 'event_role_co_occur.json')
        self.event_co_occur_embedding = self._init_embedding(event_co_occur_filename, len(hyper.event2id))

        self.meta_mask = self._init_mask()

        self.tranform_matrix = torch.zeros((hyper.role_vocab_size, hyper.n)).float().cuda(self.gpu)

        self.one_hot_embedding = nn.Embedding(hyper.n, hyper.n).cuda(self.gpu)
        nn.init.eye_(self.one_hot_embedding.weight)

    def _init_embedding(self, filename: str, embed_rows: int):
        embedding = nn.Embedding(embed_rows, self.hyper.role_vocab_size)
        embedding.weight.data.copy_(self._read_weight(filename))
        embedding.weight.requires_grad = False
        embedding = embedding.cuda(self.gpu)
        return embedding
    
    @staticmethod
    def _read_weight(filename: str) -> torch.tensor:
        key2rows = JsonHandler.read_json(filename)
        matrix = []
        for i in range(len(key2rows)):
            matrix.append(key2rows[str(i)])
        np_matrix = np.array(matrix)
        return torch.from_numpy(np_matrix)
    
    def _init_mask(self):
        meta_mask = torch.zeros((1, self.hyper.role_vocab_size)).float().cuda(self.gpu)
        meta_mask[0, self.hyper.meta_roles] = 1
        return meta_mask

    def generate_soft_label(self, sample, remap: Dict[int, int]):
        entity_type, event_type = sample.entity_type.cuda(self.gpu), sample.event_type.cuda(self.gpu)
        entity_weight = self.entity_co_occur_embedding(entity_type)
        event_weight = self.event_co_occur_embedding(event_type)
        weight = entity_weight + event_weight

        weight *= self.meta_mask

        weight = weight.matmul(self._gen_cur_tranform_matrix(remap))

        meta_label = sample.label.cuda(self.gpu)
        one_hot = self.one_hot_embedding(meta_label)
        neg_one_hot = 1 - one_hot
        weight *= neg_one_hot
        label = (1 - self.soft) * one_hot
        weight_sum = torch.sum(weight, dim=-1)
        label[weight_sum == 0] += self.soft * one_hot[weight_sum == 0]
        label[weight_sum != 0] += self.soft * weight[weight_sum != 0] / weight_sum[weight_sum != 0].unsqueeze(-1)
        label = label.detach()
        return label

    def _gen_cur_tranform_matrix(self, remap: Dict[int, int]):
        tranform_matrix = self.tranform_matrix.clone()
        for orig_idx, idx in remap.items():
            tranform_matrix[orig_idx, idx] = 1
        return tranform_matrix