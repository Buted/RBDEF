import torch
import logging

import torch.nn as nn


class AdjustCrossEntropy:
    def __init__(self, gamma: float):
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.gamma = gamma

    # def __call__(self, logits, target, adjusts):
    #     loss = self.cross_entropy(logits, target=target)
    #     adjust_loss = self.gamma * adjusts.float() * loss
    #     loss = loss.mean()

    #     adjust_num = torch.sum((adjusts > 0).float())
    #     if adjust_num > 0:
    #         adjust_loss = torch.sum(adjust_loss) / adjust_num
    #         loss += adjust_loss

    #     return loss

    def __call__(self, logits, target, adjusts):
        loss = self.cross_entropy(logits, target=target)
        # adjusts = (adjusts > 0).float()
        adjust_num = torch.sum(adjusts).item()
        # logging.info(adjust_num)
        # loss_tmp = loss.clone()
        if adjust_num > 0:
            # logging.info(adjust_num)
            reweight = target.shape[0] * self.gamma / adjust_num
            # reweight += 1
            reweight = adjusts * reweight
            reweight += 1
            reweight = reweight.detach()
            loss = loss * reweight
        # loss -= loss_tmp
        return loss.mean()