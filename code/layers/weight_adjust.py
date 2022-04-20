import os
import torch

import numpy as np
import torch.nn as nn

from code.config import Hyper
from code.utils import JsonHandler


class WeightAdjust:
    def __init__(self, hyper: Hyper):
        self.role_adjust = self._init_role_adjust_embedding(hyper)
        self.entity_adjust = self._init_entity_adjust_embedding(hyper)
        self.event_adjust = self._init_event_adjust_embedding(hyper)
    
    def _init_role_adjust_embedding(self, hyper: Hyper):
        emb = nn.Embedding(hyper.role_vocab_size, 1)
        weight = np.zeros((hyper.role_vocab_size, 1))
        for role in hyper.meta_roles:
            weight[role] = 1
        weight = torch.from_numpy(weight)
        emb.weight.data.copy_(weight)
        emb.weight.requires_grad = False
        emb.to(hyper.gpu)
        return emb

    def _init_entity_adjust_embedding(self, hyper: Hyper):
        emb = nn.Embedding(len(hyper.entity2id), 1)
        weight = np.zeros((len(hyper.entity2id), 1))
        role2entity = JsonHandler.read_json(os.path.join(hyper.data_root, 'role2entity.json'))
        for role in hyper.meta_roles:
            entities = role2entity[str(role)]
            for ent in entities:
                weight[ent] = 1
        weight = torch.from_numpy(weight)
        emb.weight.data.copy_(weight)
        emb.weight.requires_grad = False
        emb.to(hyper.gpu)
        return emb
    
    def _init_event_adjust_embedding(self, hyper: Hyper):
        emb = nn.Embedding(len(hyper.event2id), 1)
        weight = np.zeros((len(hyper.event2id), 1))
        role2event = JsonHandler.read_json(os.path.join(hyper.data_root, 'role2event.json'))
        for role in hyper.meta_roles:
            events = role2event[str(role)]
            for eve in events:
                weight[eve] = 1
        weight = torch.from_numpy(weight)
        emb.weight.data.copy_(weight)
        emb.weight.requires_grad = False
        emb.to(hyper.gpu)
        return emb
    
    def __call__(self, role, entity, event):
        # score = self.role_adjust(role) + self.entity_adjust(entity) + self.event_adjust(event)
        # score = self.entity_adjust(entity) + self.event_adjust(event)
        score = self.event_adjust(event)
        # score = self.entity_adjust(entity)
        # score = self.role_adjust(role)
        score = score.squeeze()
        score = score.detach()
        # return score / torch.sum(score)
        return score