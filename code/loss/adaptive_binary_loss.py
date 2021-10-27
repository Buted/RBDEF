import torch.nn as nn


class AdaptiveBinaryLoss(nn.Module):
    def __init__(self):
        super(AdaptiveBinaryLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, labels, logits, threshold):
        positive_logits = logits - threshold
        # negative_logits = - positive_logits
        # logits = labels * positive_logits + (1-labels) * negative_logits
        return self.loss(positive_logits, target=labels)