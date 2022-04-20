import torch

import torch.nn as nn

from typing import Dict
from functools import partial

from code.config import Hyper
from code.layers import MetaClassifier, MetaWithEmbeddingClassifier
from code.layers import Encoder
from code.models.model import Model
from code.loss import SoftCrossEntropyLoss, MaskHandler, AdjustCrossEntropy
from code.metrics import MetaF1, Accuracy
from code.layers import WeightAdjust


class MetaAEModel(Model):
    def __init__(self, hyper: Hyper):
        super(MetaAEModel, self).__init__()
        self.gpu = hyper.gpu

        self.encoder = Encoder(hyper)
        self.encoder.load()

        # self.classifier = MetaClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.n)
        self.classifier = MetaWithEmbeddingClassifier(hyper, self.encoder.embed_dim)

        self.soft_loss = SoftCrossEntropyLoss()
        self.loss = nn.CrossEntropyLoss()
        self.adjust_loss = AdjustCrossEntropy(hyper.gamma)

        self.mask_handler = MaskHandler(hyper)

        self.metric = MetaF1(hyper)
        self.get_metric = self.metric.report
        self.accuracy = Accuracy()

        self.remap = {r: i for i, r in enumerate(hyper.meta_roles)}
        self.soft = False

        self.weight_adjuster = WeightAdjust(hyper)

        # self.dropout = SampleDropout(hyper)

        self.to(self.gpu)

    def meta_forward(self, sample, classifier, remap: Dict[int, int]=None, outloop: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        # outloop = False
        with torch.no_grad():
            # if not outloop:
                # labels = self.mask_handler.generate_soft_label(sample, remap)
            entity_encoding, trigger_encoding = self.encoder(sample, False)
        entity_encoding, trigger_encoding = entity_encoding.detach(), trigger_encoding.detach()

        entity_type, event_type = sample.entity_type, sample.event_type
        logits = classifier(entity_encoding, trigger_encoding, entity_type, event_type)
        # logits = classifier(entity_encoding, trigger_encoding)

        if outloop:
            # entity_type, event_type = sample.entity_type, sample.event_type
            # original_labels = sample.ori_label
            # adjusted_weight = self.weight_adjuster(original_labels, entity_type, event_type)
            adjusted_weight = sample.important.float()
            # self.adjust_num.append(torch.sum(adjusted_weight).item())
            # if len(self.adjust_num) >= 100:
            #     print(sum(self.adjust_num)/100)
            #     exit(0)
            output['loss'] = self.adjust_loss(logits, target=labels, adjusts=adjusted_weight)
        else:
            output['loss'] = self.loss(logits, target=labels)            
            
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

        entity_type, event_type = sample.entity_type.cuda(self.gpu), sample.event_type.cuda(self.gpu)
        logits = self.classifier(entity_encoding, trigger_encoding, entity_type, event_type)

        
        if is_train:
            output['loss'] = self.soft_loss(logits, target=soft_labels) if self.soft else self.loss(logits, target=labels)
            output["description"] = partial(self.description, output=output)
        else:
            self._update_metric(logits, labels)
            output["probability"] = torch.softmax(logits, dim=-1) 

        return output

    def _update_metric(self, logits, labels) -> None:
        predicts = torch.argmax(logits, dim=-1)
        self.metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())
        self.accuracy.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())
 
    def save(self):
        self.classifier.save()
        return

    def reset(self):
        self.metric.reset()
        self.accuracy.reset()

