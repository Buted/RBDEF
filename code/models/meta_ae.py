import torch

import torch.nn as nn

from typing import Dict
from functools import partial

from code.config import Hyper
from code.models.classifier import MetaClassifier
from code.models.encoder import Encoder
from code.models.model import Model
from code.loss import SoftCrossEntropyLoss, MaskHandler
from code.metrics import MetaF1
from code.models.sample_dropout import SampleDropout


class MetaAEModel(Model):
    def __init__(self, hyper: Hyper):
        super(MetaAEModel, self).__init__()
        self.gpu = hyper.gpu

        self.encoder = Encoder(hyper)
        self.encoder.load()

        self.classifier = MetaClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.n)

        self.soft_loss = SoftCrossEntropyLoss()
        self.loss = nn.CrossEntropyLoss()

        self.mask_handler = MaskHandler(hyper)

        self.metric = MetaF1(hyper)
        self.get_metric = self.metric.report

        self.remap = {i: hyper.meta_roles.index(i) if i in hyper.meta_roles else 0 for i in range(hyper.role_vocab_size)}
        self.soft = True

        # self.dropout = SampleDropout(hyper)

        self.to(self.gpu)

    def meta_forward(self, sample, classifier, remap: Dict[int, int], is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        with torch.no_grad():
            if self.soft:
                labels = self.mask_handler.generate_soft_label(sample, remap)
            entity_encoding, trigger_encoding = self.encoder(sample, False)
        entity_encoding, trigger_encoding = entity_encoding.detach(), trigger_encoding.detach()

        logits = classifier(entity_encoding, trigger_encoding)

        output['loss'] = self.soft_loss(logits, target=labels) if self.soft else self.loss(logits, target=labels)
        
        if is_train:
            output["description"] = partial(self.description, output=output)
            
        return output

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)
        hard_label = labels

        with torch.no_grad():
            if self.soft:
                soft_labels = self.mask_handler.generate_soft_label(sample, self.remap)
            entity_encoding, trigger_encoding = self.encoder(sample, False)
        entity_encoding, trigger_encoding = entity_encoding.detach(), trigger_encoding.detach()

        # if is_train:
        #     entity_encoding = self.dropout(entity_encoding, hard_label)
        #     trigger_encoding = self.dropout(trigger_encoding, hard_label)
        logits = self.classifier(entity_encoding, trigger_encoding)

        output['loss'] = self.soft_loss(logits, target=soft_labels) if self.soft else self.loss(logits, target=labels)
        
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