import torch

from functools import partial
from typing import Dict

from code.models.meta_ae import MetaAEModel
from code.config import Hyper
from code.metrics import MetaWithOtherF1


class MetaWithOtherAEModel(MetaAEModel):
    def __init__(self, hyper: Hyper):
        super(MetaWithOtherAEModel, self).__init__(hyper)
        
        self.metric = MetaWithOtherF1(hyper)
        self.get_metric = self.metric.report

        self.remap = {i: hyper.meta_roles.index(i)+1 if i in hyper.meta_roles else 0 for i in range(hyper.role_vocab_size)}

        self.soft = False
    
    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        with torch.no_grad():
            if self.soft:
                soft_labels = self.mask_handler.generate_soft_label(sample, self.remap)
        entity_encoding, trigger_encoding = self.encoder(sample, is_train)
        # entity_encoding, trigger_encoding = entity_encoding.detach(), trigger_encoding.detach()

        logits = self.classifier(entity_encoding, trigger_encoding)

        output['loss'] = self.soft_loss(logits, target=soft_labels) if self.soft else self.loss(logits, target=labels)
        
        if is_train:
            output["description"] = partial(self.description, output=output)
        else:
            self._update_metric(logits, labels)
            output["probability"] = torch.softmax(logits, dim=-1) 

        return output


    def save(self):
        # self.classifier.save()
        return
    
    def load(self):
        self.classifier.load()