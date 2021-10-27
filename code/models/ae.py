import torch

import torch.nn as nn
import torch.hub as pretrained

from typing import Dict, Tuple
from functools import partial
# from transformers.models.bert import BertModel

from code.config import Hyper
from code.models.classifier import MainClassifier
from code.models.encoder import Encoder
from code.models.model import Model
from code.metrics import F1, Indicator


class AEModel(Model):
    def __init__(self, hyper: Hyper):
        super(AEModel, self).__init__()
        self.gpu = hyper.gpu
        
        self.encoder = Encoder(hyper)

        self.classifier = MainClassifier(self.encoder.embed_dim, hyper.role_vocab_size)

        self.main_loss = nn.CrossEntropyLoss()

        self.metric = F1(hyper)
        self.get_metric = self.metric.report
        self.role_indicator = Indicator(hyper)
        # self.NonRole_indicator = Indicator(hyper, {0: [1, 0], 1: [0, 1]})

        self.to(hyper.gpu)

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        entity_encoding, trigger_encoding = self.encoder(sample, is_train)
        logits = self.classifier(entity_encoding, trigger_encoding)

        output['loss'] = self.main_loss(logits, target=labels)
        
        if is_train:
            output["description"] = partial(self.description, output=output)
        else:
            self._update_metric(logits, labels)
            output["probability"] = torch.softmax(logits, dim=-1)
            
        return output

    def _update_metric(self, logits, labels) -> None:
        predicts = torch.argmax(logits, dim=-1)
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())
    
    def reset_indicators(self) -> None:
        self.role_indicator.reset()
        # self.NonRole_indicator.reset()

    def update_indicators(self, sample, prob):
        # to_binary = lambda x: torch.gt(x, 0).long()
        self.role_indicator.update(prob, sample.label, sample.entity_type)
        # self.NonRole_indicator.update(prob, to_binary(sample.label), to_binary(sample.entity_type))
    
    def report(self):
        return (
            self.metric.report_all(), 
            self.role_indicator.report(),
            # self.NonRole_indicator.report()
            None
        )