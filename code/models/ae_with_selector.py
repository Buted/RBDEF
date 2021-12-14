import torch

import torch.nn as nn

from typing import Dict
from functools import partial

from code.config import Hyper
from code.layers.classifier import ScaleMainClassifier, SelectorClassifier
from code.layers import Encoder
from code.models.model import Model
from code.metrics.multi_task_f1 import MultiTaskF1


class AEWithSelector(Model):
    def __init__(self, hyper: Hyper):
        super(AEWithSelector, self).__init__()
        self.gpu = hyper.gpu
        self.threshold = 0.5
        self.encoder = Encoder(hyper)

        self.main_classifier = ScaleMainClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.role_vocab_size)
        self.selector = SelectorClassifier(self.encoder.embed_dim, hyper.out_dim)
        
        self.main_loss = nn.CrossEntropyLoss()
        self.selector_loss = nn.BCEWithLogitsLoss()

        self.metric = MultiTaskF1(hyper)
        self.get_metric = self.metric.report

        self.to(hyper.gpu)

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)
        binary_labels = sample.binary_label.cuda(self.gpu).float()

        entity_encoding, trigger_encoding = self.encoder(sample, is_train)
        main_logits = self.main_classifier(entity_encoding, trigger_encoding)
        selector_logits = self.selector(entity_encoding, trigger_encoding).squeeze()

        output['loss'] = self.main_loss(main_logits, target=labels) + self.selector_loss(selector_logits, target=binary_labels)

        if is_train:
            output["description"] = partial(self.description, output=output)
        else:
            self._update_metric((main_logits, selector_logits), (labels, binary_labels))
            
        return output

    def _update_metric(self, logits, labels) -> None:
        main_logits, selector_logits = logits
        main_labels, selector_labels = labels
        main_predicts = torch.argmax(main_logits, dim=-1).cpu()
        selector_predicts = torch.gt(torch.sigmoid(selector_logits), self.threshold).int().cpu()
        self.metric.update(golden_labels=(main_labels.cpu(), selector_labels.cpu()), predict_labels=(main_predicts, selector_predicts))
    
    def save(self):
        return