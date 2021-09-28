import torch

from typing import Dict, List

from code.config import Hyper


class Indicator:
    def __init__(self, hyper: Hyper, co_occur_matrix: Dict[int, List[int]] = None):
        self.hyper = hyper
        self.co_occur_matrix = co_occur_matrix or hyper.co_occur_matrix
        self.reset()

    def reset(self):
        empty_dict = lambda: {i: 0 for i in range(self.hyper.role_vocab_size)}
        self.role2golden_num = empty_dict()
        self.role2golden_prob = empty_dict()
        self.role2unrelated_num = empty_dict()
        self.role2unrelated_prob = empty_dict()
        self.role2inditator = empty_dict()

    def update(self, prob: torch.tensor, role: torch.tensor, entity_type: torch.tensor) -> None:
        for e, r, p in zip(entity_type, role, prob):
            e, r = e.item(), r.item()
            self.role2golden_num[r] += 1
            self.role2golden_prob[r] += p[r].item()
            for i, occur in enumerate(self.co_occur_matrix[e]):
                if occur == 0:
                    self.role2unrelated_num[i] += 1
                    self.role2unrelated_prob[i] += p[i].item()

    def report(self):
        for role in self.role2golden_num:
            self.role2golden_prob[role] /= self.role2golden_num[role] or 1
            self.role2unrelated_prob[role] /= self.role2unrelated_num[role] or 1
            self.role2inditator[role] = self.role2golden_prob[role] + 1 - self.role2unrelated_prob[role]
        
        return [
            {
                "golden": self.role2golden_prob[role],
                "unrelated": self.role2unrelated_prob[role],
                "indicator": self.role2inditator[role]
            }
            for role in self.role2inditator
        ]