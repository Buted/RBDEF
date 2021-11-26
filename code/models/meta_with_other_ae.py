from code.models.meta_ae import MetaAEModel
from code.config import Hyper
from code.metrics import MetaWithOtherF1


class MetaWithOtherAEModel(MetaAEModel):
    def __init__(self, hyper: Hyper):
        super(MetaWithOtherAEModel, self).__init__(hyper)
        
        self.metric = MetaWithOtherF1(hyper)
        self.get_metric = self.metric.report

        self.remap = {i: hyper.meta_roles.index(i)+1 if i in hyper.meta_roles else 0 for i in range(hyper.role_vocab_size)}

        # self.soft = False
    
    def save(self):
        return