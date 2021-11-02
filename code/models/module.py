import os
import torch

import torch.nn as nn


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.model_dir = os.path.join('saved_models', 'layers')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
    
    def save(self):
        torch.save(
            self.state_dict(),
            os.path.join(self.model_dir, self.__class__.__name__),
        )

    def load(self):
        self.load_state_dict(
            torch.load(os.path.join(self.model_dir, self.__class__.__name__))
        )