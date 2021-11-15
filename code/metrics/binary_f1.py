from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

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
        report = classification_report(
            y_true=self.goldens, y_pred=self.predicts, 
            labels=[0, 1], 
            digits=4, output_dict=True, zero_division=0
        )
        # print(precision, recall, f1_score, support)
        return {
            'accuarcy': accuracy,
            'fscore': accuracy,
            'avg fscore': report["macro avg"]["f1-score"],
            '0-pre': report["0"]["precision"],
            '0-recall': report["0"]["recall"],
            '0-fscore': report["0"]["f1-score"],
            '1-pre': report["1"]["precision"],
            '1-recall': report["1"]["recall"],
            '1-fscore': report["1"]["f1-score"]
        }
    
    def report_all(self):
        accuracy = accuracy_score(y_pred=self.predicts, y_true=self.goldens)
        report = classification_report(
            y_true=self.goldens, y_pred=self.predicts, 
            labels=[0, 1], 
            digits=4, output_dict=True, zero_division=0
        )
        
        output_keys = ["macro avg"] + ["0", "1"]

        return [{'accuarcy': accuracy}] + \
        [
            {
                'precision': report[key]["precision"],
                'recall': report[key]["recall"], 
                'fscore': report[key]["f1-score"],
                'support': report[key]["support"]
            }
            for key in output_keys
        ]


    def reset(self):
        self.predicts = []
        self.goldens = []