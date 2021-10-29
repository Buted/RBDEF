import torch

import torch.nn as nn

from typing import Dict
from functools import partial

from code.config import Hyper
from code.models.classifier import MetaClassifier
from code.models.encoder import Encoder
from code.models.model import Model
from code.metrics import MetaF1
# from code.metrics import F1


class MetaAEModel(Model):
    def __init__(self, hyper: Hyper):
        super(MetaAEModel, self).__init__()
        self.gpu = hyper.gpu

        self.encoder = Encoder(hyper)
        self.encoder.load()

        self.classifier = MetaClassifier(self.encoder.embed_dim, hyper.out_dim)

        self.loss = nn.CrossEntropyLoss()
        
        self.metric = MetaF1(hyper)
        # self.metric = F1(hyper)
        # self.metric.valid_labels = list(range(10))
        self.get_metric = self.metric.report

        self.to(self.gpu)

    def meta_forward(self, sample, classifier, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        with torch.no_grad():
            entity_encoding, trigger_encoding = self.encoder(sample, is_train)
        entity_encoding, trigger_encoding = entity_encoding.detach(), trigger_encoding.detach()

        logits = classifier(entity_encoding, trigger_encoding)

        output['loss'] = self.loss(logits, target=labels)
        
        if is_train:
            output["description"] = partial(self.description, output=output)
            
        return output

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        entity_encoding, trigger_encoding = self.encoder(sample, is_train)
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
        # return