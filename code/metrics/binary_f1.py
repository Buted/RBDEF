from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
import numpy as np


class BinaryMetric:
    def __init__(self):
        self.predicts = []
        self.goldens = []

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

        self.predicts.extend(predict_labels)
        self.goldens.extend(golden_labels)
    
    def get_metric(self):
        accuracy = accuracy_score(y_pred=self.predicts, y_true=self.goldens)
        # print(precision, recall, f1_score, support)
        return {
            'accuarcy': accuracy,
            'fscore': accuracy
        }

    def reset(self):
        self.predicts = []
        self.goldens = []