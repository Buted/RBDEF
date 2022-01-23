import torch
import logging

from typing import Dict, Tuple
from functools import reduce

from code.config import Hyper
from code.layers import ScaleHeadClassifier, SelectorClassifier
from code.layers import Encoder
from code.models.model import Model
from code.metrics.roc import ROCCurve


class Routing(Model):
    def __init__(self, hyper: Hyper):
        super(Routing, self).__init__()
        self.gpu = hyper.gpu

        self.encoder = Encoder(hyper)
        self.selectors = SelectorClassifier.load_group(self.encoder.embed_dim, hyper.out_dim)
        self.head_classifier = ScaleHeadClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.role_vocab_size - hyper.n + 1)
        self.load()

        self.metric = ROCCurve()
        self.get_metric = self.metric.report

        self.to(hyper.gpu)

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        entity_encoding, trigger_encoding = self.encoder(sample, False)

        select_logits = self.selector(entity_encoding, trigger_encoding)

        head_logits = self.head_classifier(entity_encoding, trigger_encoding)
        
        self._update_metric((select_logits, head_logits), labels)
            
        return output

    def selector(self, entity_encoding, trigger_encoding):
        return (selector(entity_encoding, trigger_encoding).squeeze() for selector in self.selectors)

    def _update_metric(self, logits, labels) -> None:
        predicts = self._generate_predicts(logits)
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts)

    def _generate_predicts(self, logits: Tuple):
        select_logits, head_logits = logits    

        head_prob = torch.sigmoid(head_logits)

        neg_select_prob, best_select_prob = self.generate_choice(select_logits)

        head_routing = head_prob[:, 1].clone()
        head_prob[:, 1] = 0
        head_routing = head_routing / (torch.max(head_prob, dim=-1)[0] + head_routing)

        double_routing = neg_select_prob + (1 - neg_select_prob) * best_select_prob
        triple_routing = double_routing + (1 - double_routing) * head_routing
        
        return (best_select_prob.cpu(), head_routing.cpu(), triple_routing.cpu(), double_routing.cpu())

    def generate_choice(self, select_logits):
        def choice(logit):
            return torch.sigmoid(logit)
        
        choices = [choice(logit) for logit in select_logits]
        return choices[-1], choices[1]

    def plot(self):
        self.metric.plot()

    def load(self):
        self.encoder.load()
        self.head_classifier.load()

    def save(self):
        return