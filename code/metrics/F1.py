import torch

import numpy as np

from typing import Dict, List
from sklearn.metrics import classification_report

from code.config import Hyper, NonRole

class F1:
    def __init__(self, hyper: Hyper):
        self.reset()
        labels = list(hyper.id2role.keys())
        labels.remove(hyper.role2id[NonRole])
        self.valid_labels = labels

    def reset(self) -> None:
        self.golden = []
        self.predict = []
    
    def update(self, golden_labels, predict_labels) -> None:
        golden_labels = self._to_list(golden_labels)
        predict_labels = self._to_list(predict_labels)
        
        assert len(golden_labels) == len(predict_labels),  "predicted labels should be as long as golden labels"
        
        self.golden.extend(golden_labels)
        self.predict.extend(predict_labels)

    @staticmethod
    def _to_list(labels) -> List[int]:
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        if isinstance(labels, np.ndarray):
            shape = labels.shape
            if len(shape) > 1:
                labels = labels.view(-1)
            labels = labels.tolist()
        if not isinstance(labels, list):
            raise ValueError("Wrong type of labels: %s" % type(labels))
        return labels

    def report(self) -> Dict[str, float]:
        report = classification_report(
            y_true=self.golden, y_pred=self.predict, 
            labels=self.valid_labels, 
            digits=4, output_dict=True, zero_division=0
        )["macro avg"]
        
        return {
            'precision': report["precision"],
            'recall': report["recall"], 
            'fscore': report["f1-score"],
            'support': report["support"]
        }
    
    def report_all(self) -> List[Dict]:
        report = classification_report(
            y_true=self.golden, y_pred=self.predict, 
            labels=self.valid_labels, 
            digits=4, output_dict=True, zero_division=0
        )
        
        output_keys = ['micro avg'] + [str(i) for i in self.valid_labels]
        
        return [
            {
                'precision': report[key]["precision"],
                'recall': report[key]["recall"], 
                'fscore': report[key]["f1-score"],
                'support': report[key]["support"]
            }
            for key in output_keys
        ]
