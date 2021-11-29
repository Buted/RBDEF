import torch
import logging

import torch.nn as nn

from typing import Dict
from functools import partial

from code.config import Hyper
from code.models.classifier import ScaleHeadClassifier
from code.models.encoder import Encoder
from code.models.model import Model
from code.metrics import HeadF1
from tests.time import log_time


class HeadAEModel(Model):
    def __init__(self, hyper: Hyper):
        super(HeadAEModel, self).__init__()
        self.gpu = hyper.gpu
        
        self.encoder = Encoder(hyper)
        self.encoder.load()

        self.classifier = ScaleHeadClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.role_vocab_size - len(hyper.meta_roles) + 1)

        self.loss = nn.CrossEntropyLoss()

        self.metric = HeadF1(hyper)
        self.get_metric = self.metric.report

        self.to(hyper.gpu)

    # @log_time
    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        with torch.no_grad():
            entity_encoding, trigger_encoding = self.encoder(sample, False)
        entity_encoding, trigger_encoding = entity_encoding.detach(), trigger_encoding.detach()

        logits = self.classifier(entity_encoding, trigger_encoding)

        output['loss'] = self.loss(logits, target=labels)
        
        if is_train:
            output["description"] = partial(self.description, output=output)
        else:
            self._update_metric(logits, labels)
            output["probability"] = torch.softmax(logits, dim=-1)
            
        return output

    def _update_metric(self, logits, labels) -> None:
        predicts = torch.argmax(logits, dim=-1)
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())
    
    def save(self):
        self.classifier.save()
        return
