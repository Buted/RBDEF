import torch

import numpy as np

from code.metrics.binary_f1 import BinaryMetric


class MergeF1(BinaryMetric):
    def update(self, golden_labels, predict_labels):
        def to_list(labels):
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            return labels

        predict_labels = to_list(predict_labels)
        golden_labels = to_list(golden_labels)

        assert len(predict_labels) == len(golden_labels), "predicted labels should be as long as golden labels"

        predict_labels = [1 if label == 2 else 0 for label in predict_labels]
        golden_labels = [1 if label == 2 else 0 for label in golden_labels]

        self.predicts.extend(predict_labels)
        self.goldens.extend(golden_labels)