import torch

from typing import Dict, Tuple
from functools import reduce

from code.config import Hyper
from code.layers import ScaleHeadClassifier, MetaClassifier, SelectorClassifier
from code.layers import Encoder
from code.models.model import Model
from code.metrics import F1


class FusedAEModel(Model):
    def __init__(self, hyper: Hyper):
        super(FusedAEModel, self).__init__()
        self.gpu = hyper.gpu
        self.meta_roles = hyper.meta_roles
        self.head_roles = [0, 1] + [i for i in range(1, hyper.role_vocab_size) if i not in self.meta_roles]
        self.threshold = 0.5
        
        self.encoder = Encoder(hyper)
        self.selectors = SelectorClassifier.load_group(self.encoder.embed_dim, hyper.out_dim)
        self.head_classifier = ScaleHeadClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.role_vocab_size - hyper.n + 1)
        self.meta_classifier = MetaClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.n)
        self.load()

        self.metric = F1(hyper)
        self.get_metric = self.metric.report

        self.to(hyper.gpu)

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        entity_encoding, trigger_encoding = self.encoder(sample, False)

        select_logits = self.selector(entity_encoding, trigger_encoding)

        head_logits = self.head_classifier(entity_encoding, trigger_encoding)
        meta_logits = self.meta_classifier(entity_encoding, trigger_encoding)

        
        self._update_metric((select_logits, head_logits, meta_logits), labels)
            
        return output

    def selector(self, entity_encoding, trigger_encoding):
        return (selector(entity_encoding, trigger_encoding).squeeze() for selector in self.selectors)

    def _update_metric(self, logits, labels) -> None:
        predicts = self._generate_predicts(logits, labels)
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())

    def _generate_predicts(self, logits: Tuple, labels):
        select_logits, head_logits, meta_logits = logits    

        head_predicts = torch.argmax(head_logits, dim=-1)
        meta_predicts = torch.argmax(meta_logits, dim=-1)

        select_predicts = self.generate_choice(select_logits)

        predicts = torch.zeros_like(meta_predicts)
        for i, (select_p, head_p, meta_p) in enumerate(zip(select_predicts, head_predicts, meta_predicts)):

            if select_p > 1 or head_p == 1:
                predicts[i] = self.meta_roles[meta_p]
            else:
                predicts[i] = self.head_roles[head_p]

        return predicts

    def generate_choice(self, select_logits):
        def choice(logit):
            select_prob = torch.sigmoid(logit)
            select_predicts = torch.gt(select_prob, self.threshold).int()
            return select_predicts
        
        choices = [choice(logit) for logit in select_logits]
        final_choice = reduce(lambda x, y: x + y, choices)
        return final_choice

    def save(self):
        return

    def load(self):
        self.encoder.load()
        self.head_classifier.load()
        self.meta_classifier.load()