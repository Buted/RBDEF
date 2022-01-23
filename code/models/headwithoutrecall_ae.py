from code.models.head_ae import HeadAEModel
from code.config import Hyper
from code.layers.classifier import ScaleHeadWithoutRecallClassifier


class HeadWithoutRecallAEModel(HeadAEModel):
    def __init__(self, hyper: Hyper):
        super(HeadWithoutRecallAEModel, self).__init__(hyper)
        self.classifier = ScaleHeadWithoutRecallClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.role_vocab_size - len(hyper.meta_roles))
        self.to(hyper.gpu)
