from typing import Dict, List
from sklearn.metrics import classification_report

from code.metrics.F1 import F1
from code.config import Hyper


class MetaF1(F1):
    def __init__(self, hyper: Hyper):
        super(MetaF1, self).__init__(hyper)
        self.valid_labels = list(range(1, 10))
    
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
        
        output_keys = ['macro avg'] + [str(i) for i in self.valid_labels]
        
        return [
            {
                'precision': report[key]["precision"],
                'recall': report[key]["recall"], 
                'fscore': report[key]["f1-score"],
                'support': report[key]["support"]
            }
            for key in output_keys
        ]