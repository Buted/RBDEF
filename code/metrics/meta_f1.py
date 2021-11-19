from typing import Dict, List
from sklearn.metrics import precision_recall_fscore_support

from code.metrics.F1 import F1
from code.config import Hyper


class MetaF1(F1):
    def __init__(self, hyper: Hyper):
        super(MetaF1, self).__init__(hyper)
        self.valid_labels = list(range(hyper.n))
        self.use_micro = hyper.metric == 'micro'
        if self.use_micro:
            self.metric = 'macro'
    
    def report(self) -> Dict[str, float]:
        if not self.use_micro:
            return super().report()
        p, r, f1, support = precision_recall_fscore_support(
            y_true=self.golden, y_pred=self.predict,
            labels=self.valid_labels,
            average='micro', zero_division=0
        )

        return {
            'precision': p,
            'recall': r, 
            'fscore': f1,
            'support': len(self.golden)
        }

    def report_all(self) -> List[Dict]:
        report = super().report_all()
        if self.use_micro:
            p, r, f1, support = precision_recall_fscore_support(
            y_true=self.golden, y_pred=self.predict,
            labels=self.valid_labels,
            average='micro', zero_division=0
            )
            report[0] = {
                'precision': p,
                'recall': r, 
                'fscore': f1,
                'support': len(self.golden)
            }
        return report