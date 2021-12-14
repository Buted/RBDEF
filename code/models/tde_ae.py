import torch

import torch.nn as nn

from typing import Dict
from functools import partial

from code.models.model import Model
from code.config import Hyper
from code.layers import TDEClassifier
from code.layers import Encoder
from code.metrics import F1


class TDEAEModel(Model):
    def __init__(self, hyper: Hyper):
        super(TDEAEModel, self).__init__()
        self.gpu = hyper.gpu
        
        self.encoder = Encoder(hyper)

        self.classifier = TDEClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.role_vocab_size, alpha=hyper.beta)

        self.main_loss = nn.CrossEntropyLoss()

        self.metric = F1(hyper)
        self.get_metric = self.metric.report

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
            factual_logits = self.classifier.train_forward(entity_encoding, trigger_encoding)
            self._update_metric((logits, factual_logits), labels)
            output["probability"] = torch.softmax(logits, dim=-1)
            
        return output

    def _update_metric(self, logits, labels) -> None:
        predicts = self._generate_predicts(logits)
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())
    
    def _generate_predicts(self, logits) -> torch.tensor:
        counterfactual_logits, factual_logits = logits
        q = torch.softmax(counterfactual_logits, dim=-1)
        p = torch.softmax(factual_logits, dim=-1)
        q = q * (1 - p[:, 0].unsqueeze(-1)) / q[:, 0].unsqueeze(-1)
        q[:, 0] = p[:, 0]

        return torch.argmax(q, dim=-1)

    def save(self):
        return
    
    def load(self):
        return