from sklearn.metrics import precision_recall_fscore_support

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
    
    def report(self):
        precison, recall, f1_score, support =  precision_recall_fscore_support(y_pred=self.predicts, y_true=self.goldens, labels=[1])
        return {
            'precision': precison[0],
            'recall': recall[0], 
            'f1_score': f1_score[0],
            'support': support[0]
        }

    def reset(self):
        self.predicts = []
        self.goldens = []