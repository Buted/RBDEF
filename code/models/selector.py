import torch

import torch.nn as nn

from typing import Dict, Tuple
from functools import partial

from code.config import Hyper
from code.models.classifier import SelectorClassifier
from code.models.encoder import Encoder
from code.models.model import Model
from code.metrics.binary_f1 import BinaryMetric
from code.metrics.pr_curve import BinaryPRCurve


class Selector(Model):
    def __init__(self, hyper: Hyper):
        super(Selector, self).__init__()
        self.gpu = hyper.gpu
        self.threshold = 0.5
        self.encoder = Encoder(hyper)
        self.encoder.load()

        self.classifier = SelectorClassifier(self.encoder.embed_dim, hyper.out_dim)
        self.classifier.load_from_meta(hyper.n)

        self.loss = nn.BCEWithLogitsLoss()

        self.metric = BinaryMetric()
        self.curve = BinaryPRCurve()
        self.get_metric = self.metric.get_metric
        self.to(self.gpu)

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu).float()

        with torch.no_grad():
            entity_encoding, trigger_encoding = self.encoder(sample, False)
        entity_encoding, trigger_encoding = entity_encoding.detach(), trigger_encoding.detach()

        logits = self.classifier(entity_encoding, trigger_encoding)
        logits = logits.squeeze()

        output['loss'] = self.loss(logits, target=labels)
        # output['loss']  += 0.02 * self._polar_constraint(logits)
        
        if is_train:
            output["description"] = partial(self.description, output=output)
        else:
            self._update_metric(logits, labels)
            output["probability"] = torch.sigmoid(logits)
            self.curve.update(golden_labels=labels.cpu(), predict_labels=output["probability"].cpu())
            
        return output

    def _update_metric(self, logits, labels) -> None:
        predicts = torch.gt(torch.sigmoid(logits), self.threshold).int()
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())
    
    def save(self):
        # self.classifier.save()
        return

    @staticmethod
    def _polar_constraint(logits):
        prob = torch.sigmoid(logits)
        batch_size = prob.size()[0]
        prob_unipolar = 1 - prob

        mean = torch.mean(prob)
        deviation = prob - mean
        constraint_centrifugal = torch.abs(deviation)

        polar_constraint = prob_unipolar - constraint_centrifugal
        return torch.sum(polar_constraint) / batch_size