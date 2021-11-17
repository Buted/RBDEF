from code.config import Hyper
from code.models.classifier import SimpleClassifier
from code.models.ae import AEModel


class SimpleAEModel(AEModel):
    def __init__(self, hyper: Hyper):
        super(SimpleAEModel, self).__init__(hyper)

        self.classifier = SimpleClassifier(self.encoder.embed_dim, hyper.role_vocab_size)

        self.to(hyper.gpu)
    
    def save(self):
        self.encoder.save()