import torch

import torch.nn as nn

from typing import Dict, List, Tuple
from functools import partial

from code.config import Hyper
from code.models.classifier import MainClassifier, MetaClassifier, SelectorClassifier
from code.models.encoder import Encoder
from code.models.model import Model
from code.metrics import F1


class RecallAEModel(Model):
    def __init__(self, hyper: Hyper):
        super(RecallAEModel, self).__init__()
        self.gpu = hyper.gpu
        self.meta_roles = hyper.meta_roles
        self.meta_mask = torch.ones(hyper.role_vocab_size).float().to(self.gpu)
        for role in hyper.meta_roles:
            self.meta_mask[role] = 0
        self.threshold = 0.9
        
        self.encoder = Encoder(hyper)
        self.main_classifier = MainClassifier(self.encoder.embed_dim, hyper.role_vocab_size)
        self.meta_classifier = MetaClassifier(self.encoder.embed_dim, hyper.out_dim)
        self.selector = SelectorClassifier(self.encoder.embed_dim, hyper.out_dim)
        self.load()

        self.loss = nn.CrossEntropyLoss()

        self.metric = F1(hyper)
        self.get_metric = self.metric.report

        self.to(hyper.gpu)

    def freeze(self):
        self.encoder.freeze()
        self.main_classifier.freeze()

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)
        meta_labels = sample.meta_label.cuda(self.gpu)

        entity_encoding, trigger_encoding = self.encoder(sample, is_train)
        main_logits = self.main_classifier(entity_encoding, trigger_encoding)
        meta_logits = self.meta_classifier(entity_encoding, trigger_encoding)
        selector_logits = self.selector(entity_encoding, trigger_encoding)
        
        output['loss'] = self.loss(meta_logits, meta_labels)
        
        if is_train:
            output["description"] = partial(self.description, output=output)
        else:
            self._update_metric((main_logits, meta_logits, selector_logits), labels)
            
        return output

    def _update_metric(self, logits: Tuple, labels) -> None:
        predicts = self._generate_predicts(logits)
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())
    
    def _generate_predicts(self, logits: Tuple):
        main_logits, meta_logits, selector_logits = logits

        main_predicts = torch.argmax(main_logits, dim=-1)
        meta_predicts = torch.argmax(meta_logits, dim=-1)
        select_prob = torch.sigmoid(selector_logits)
        select_predicts = torch.gt(select_prob, self.threshold).int()

        predicts = torch.zeros_like(meta_predicts)

        for i, (main_p, meta_p, select_p) in enumerate(zip(main_predicts, meta_predicts, select_predicts)):
            branch_sum = select_p
            branch_sum += 1 if main_p in self.meta_roles else 0
            branch_sum += 0 if meta_p == 0 else 1
            choice_meta = branch_sum > 1

            if choice_meta:
                meta_logit = torch.softmax(meta_logits[i][1:], dim=-1)
                meta_p = torch.argmax(meta_logit, dim=-1)
                p = self.meta_roles[meta_p]
            else:
                main_logit = torch.softmax(main_logits[i], dim=-1)
                main_logit = torch.softmax(main_logit * self.meta_mask, dim=-1)
                p = torch.argmax(main_logit, dim=-1)
            predicts[i] = p
        return predicts

    def load(self):
        self.encoder.load()
        self.main_classifier.load()
        self.meta_classifier.load()
        self.selector.load()

    def save(self):
        return