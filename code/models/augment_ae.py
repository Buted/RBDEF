from code.config import Hyper
from code.models.ae import AEModel
from code.layers import AugmentMetaClassifier


class AugmentMetaAEModel(AEModel):
    def __init__(self, hyper: Hyper):
        super(AugmentMetaAEModel, self).__init__(hyper)
        self.encoder.load()
        self.classifier = AugmentMetaClassifier(self.encoder.embed_dim, hyper.out_dim, hyper.role_vocab_size)
        self.classifier.load_from_meta(hyper.meta_roles, self.encoder.embed_dim)

        self.to(hyper.gpu)

    def save(self):
        return