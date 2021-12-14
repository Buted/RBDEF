import torch

from typing import Dict, List
from functools import partial

from code.models.model import Model
from code.config import Hyper
from code.models.encoder import Encoder
from code.models.fair_classifier import FairClassifier
from code.loss import FairLDAMLoss
from code.metrics import F1


class FairAEModel(Model):
    def __init__(self, hyper: Hyper):
        super(FairAEModel, self).__init__()
        self.gpu = hyper.gpu
        
        self.encoder = Encoder(hyper)

        self.classifier = FairClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.role_vocab_size)

        hyper.get_role2num()
        hyper.get_group2id()
        self.main_loss = FairLDAMLoss(device=self.gpu, cls_num_list=self._gen_role_num(hyper.role2num), group_size=hyper.group_size, rho=hyper.beta)

        self.metric = F1(hyper)
        self.get_metric = self.metric.report

        self.to(hyper.gpu)
    
    @staticmethod
    def _gen_role_num(role2num: Dict[int, int]) -> List[int]:
        return [role2num[r] for r in range(len(role2num))]
        
    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)
        groups = sample.group.cuda(self.gpu)

        entity_encoding, trigger_encoding = self.encoder(sample, is_train)
        logits = self.classifier(entity_encoding, trigger_encoding)

        output['loss'] = self.main_loss(logits, target=labels, group=labels)
        
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
        return
    
    def load(self):
        return