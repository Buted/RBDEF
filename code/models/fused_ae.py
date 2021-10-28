import torch

import torch.nn as nn

from typing import Dict, List
from functools import partial

from code.config import Hyper
from code.models.classifier import MainClassifier, MetaClassifier
from code.models.encoder import Encoder
from code.models.model import Model
from code.metrics import F1

import logging

class FusedAEModel(Model):
    def __init__(self, hyper: Hyper):
        super(FusedAEModel, self).__init__()
        self.gpu = hyper.gpu
        
        self.encoder = Encoder(hyper)
        self.main_classifier = MainClassifier(self.encoder.embed_dim, hyper.role_vocab_size)
        self.meta_classifier = MetaClassifier(self.encoder.embed_dim, hyper.out_dim)
        self.load()

        self.fusing_mask = self._get_fusing_mask(hyper.role_vocab_size, hyper.meta_roles)
        # self.alpha = hyper.alpha
        self.alpha = nn.Parameter(torch.tensor(hyper.alpha), requires_grad=False)
        # self.alpha = hyper.alpha

        self.loss = nn.NLLLoss()
        # self.loss = nn.CrossEntropyLoss()

        self.metric = F1(hyper)
        self.get_metric = self.metric.report

        self.to(hyper.gpu)

    def _get_fusing_mask(self, roles_size: int, meta_roles: List[int]):
        meta_roles_size = len(meta_roles)
        mask = torch.zeros((meta_roles_size, roles_size)).float().cuda(self.gpu)
        mask.requires_grad = False
        mask.log_softmax
        for i, role_idx in enumerate(meta_roles):
            mask[i, role_idx] = 1
        return mask

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        entity_encoding, trigger_encoding = self.encoder(sample, is_train)
        main_logits = self.main_classifier(entity_encoding, trigger_encoding)
        entity_encoding, trigger_encoding = entity_encoding.detach(), trigger_encoding.detach()
        meta_logits = self.meta_classifier(entity_encoding, trigger_encoding)

        prob = self._fuse_to_prob(main_logits, meta_logits)
        
        # prob = self.alpha * main_logits + (1 - self.alpha) * torch.matmul(meta_logits, self.fusing_mask)
        output['loss'] = self._loss(prob, labels)
        # print(output['loss'])
        
        if is_train:
            output["description"] = partial(self.description, output=output)
        else:
            self._update_metric(prob, labels)
            output["probability"] = prob
            
        return output

    def _fuse_to_prob(self, main_logits, meta_logits):
        episilon = 1e-10
        prob = lambda x: torch.min(torch.softmax(x, dim=-1) + episilon, torch.full_like(x, 1))
        # main_logits -= torch.max(main_logits, dim=-1)[0].view(-1, 1)
        # meta_logits -= torch.max(meta_logits, dim=-1)[0].view(-1, 1)
        main_prob = prob(main_logits)
        meta_prob = prob(meta_logits)
        meta_prob = torch.matmul(meta_prob, self.fusing_mask)
        alpha = torch.sigmoid(self.alpha)
        prob = alpha * main_prob + (1 - alpha) * meta_prob
        return prob

    def _loss(self, prob, labels):
        log_prob = torch.log(prob)
        return self.loss(log_prob, target=labels)

    def _update_metric(self, prob, labels) -> None:
        predicts = torch.argmax(prob, dim=-1)
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())
    
    def save(self):
        logging.info(self.alpha)
        return

    def load(self):
        self.encoder.load()
        self.main_classifier.load()
        self.meta_classifier.load()