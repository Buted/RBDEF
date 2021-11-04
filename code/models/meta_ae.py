import torch

from typing import Dict
from functools import partial

from code.config import Hyper
from code.models.classifier import MetaClassifier
from code.models.encoder import Encoder
from code.models.model import Model
from code.loss.soft_cross_entropy import SoftCrossEntropyLoss
from code.loss.mask_handler import MaskHandler


class MetaAEModel(Model):
    def __init__(self, hyper: Hyper):
        super(MetaAEModel, self).__init__()
        self.gpu = hyper.gpu

        self.encoder = Encoder(hyper)
        self.encoder.load()

        self.classifier = MetaClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.n)

        self.loss = SoftCrossEntropyLoss()

        self.mask_handler = MaskHandler(hyper)

        self.to(self.gpu)

    def meta_forward(self, sample, classifier, remap: Dict[int, int], is_train: bool=False) -> Dict:
        output = {}
        # labels = sample.label.cuda(self.gpu)

        with torch.no_grad():
            labels = self.mask_handler.generate_soft_label(sample, remap)
            entity_encoding, trigger_encoding = self.encoder(sample, False)
        entity_encoding, trigger_encoding = entity_encoding.detach(), trigger_encoding.detach()

        logits = classifier(entity_encoding, trigger_encoding)

        output['loss'] = self.loss(logits, target=labels)
        
        if is_train:
            output["description"] = partial(self.description, output=output)
            
        return output

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        entity_encoding, trigger_encoding = self.encoder(sample, is_train)
        entity_encoding, trigger_encoding = entity_encoding.detach(), trigger_encoding.detach()

        logits = self.classifier(entity_encoding, trigger_encoding)

        output['loss'] = self.loss(logits, target=labels)
        
        if is_train:
            output["description"] = partial(self.description, output=output)
            
        return output

    def save(self):
        self.classifier.save()
        # return