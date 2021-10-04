import torch

import torch.nn as nn

from typing import Dict, Tuple
from functools import partial

from code.config import Hyper
from code.models.classifier import SelectorClassifier
from code.models.encoder import Encoder
from code.metrics.binary_f1 import BinaryMetric


class Selector(nn.Module):
    def __init__(self, hyper: Hyper):
        super(Selector, self).__init__()
        self.threshold = 0.5
        self.encoder = Encoder(hyper)
        self.encoder.load()

        self.classifier = SelectorClassifier(self.encoder.embed_dim)

        self.loss = nn.BCEWithLogitsLoss()

        self.metric = BinaryMetric()

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
        predicts = torch.gt(torch.sigmoid(logits), self.threshold).int()
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())
    
    def save(self):
        self.classifier.save()