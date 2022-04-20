import os
import torch
import logging

from typing import Dict, Tuple
from functools import reduce

from code.config import Hyper
from code.models.model import Model
from code.models.head_ae import HeadAEModel
from code.models.meta_ae import MetaAEModel
from code.models.selector import Selector
from code.metrics import F1


class FusedAEWithMultiEncoders(Model):
    def __init__(self, hyper: Hyper):
        super(FusedAEWithMultiEncoders, self).__init__()
        self.gpu = hyper.gpu
        self.meta_roles = hyper.meta_roles
        self.head_roles = [0, 1] + [i for i in range(1, hyper.role_vocab_size) if i not in self.meta_roles]
        # self.head_roles = [i for i in range(hyper.role_vocab_size) if i not in self.meta_roles]
        self.threshold = 0.5

        self.selectors = [Selector(hyper) for _ in range(4)]
        self.head_expert = HeadAEModel(hyper)
        self.tail_expert = MetaAEModel(hyper)
        self.load()

        self.metric = F1(hyper)
        self.get_metric = self.metric.report

        # self.to(self.gpu)
        # self = self.cpu()

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        selector_prob = (selector(sample)["probability"] for selector in self.selectors)
        # logging.info('head')
        head_prob = self.head_expert(sample)["probability"]
        # logging.info('tail')
        meta_prob = self.tail_expert(sample)["probability"]

        self._update_metric((selector_prob, head_prob, meta_prob), labels)

        return output

    def _update_metric(self, probs, labels) -> None:
        predicts = self._generate_predicts(probs, labels)
        # logging.info(predicts.shape)
        # logging.info(labels.shape)
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())

    def _generate_predicts(self, prob: Tuple, labels):
        select_prob, head_prob, tail_prob = prob

        head_predicts = torch.argmax(head_prob, dim=-1)
        # logging.info(head_predicts.shape)
        tail_predicts = torch.argmax(tail_prob, dim=-1)
        # logging.info(tail_predicts.shape)
        select_predicts = self._generate_choice(select_prob)

        predicts = torch.zeros_like(tail_predicts)
        # logging.info(predicts.shape)
        # logging.info(predicts.device)
        for i, (select_p, head_p, meta_p) in enumerate(zip(select_predicts, head_predicts, tail_predicts)):

            if select_p > 1 or head_p == 1:
                predicts[i] = self.meta_roles[meta_p]
            else:
                predicts[i] = self.head_roles[head_p]

        return predicts

    def _generate_choice(self, select_probs):
        def choice(prob):
            select_predicts = torch.gt(prob, self.threshold).int()
            return select_predicts
        
        choices = [choice(prob) for prob in select_probs]
        # return choices[1]
        final_choice = reduce(lambda x, y: x + y, choices)
        return final_choice

    def load(self):
        self._load_model(self.selectors[0], 'acc')
        self._load_model(self.selectors[1], 'avg')
        self._load_model(self.selectors[2], 'acc')
        self._load_model(self.selectors[3], 'avg')
        self._load_model(self.head_expert, 'best')
        self._load_model(self.tail_expert, 'best')
    
    def save(self):
        return
    
    def _load_model(self, model: Model, name: str):
        model.load_state_dict(
            torch.load(os.path.join('saved_models/models', model.__class__.__name__ + "_" + name))
        )