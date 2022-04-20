import torch.nn as nn

from code.layers import Module


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder: Module
        self.classifier: Module
        self.metric = None
        self.get_metric = None

    def reset(self) -> None:
        self.metric.reset()

    @staticmethod
    def description(epoch, epoch_num, output) -> str:
        return "L: {:.2f}, epoch: {}/{}:".format(
            output["loss"].item(), epoch, epoch_num,
        )    

    def save(self):
        self.encoder.save()
        self.classifier.save()

    def load(self):
        self.encoder.load()
        self.classifier.load()

    def get_parameter_number(self):
        return sum(p.numel() for p in self.parameters()) / 1e6 # magnitude using M