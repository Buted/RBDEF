from sklearn.metrics import accuracy_score

from code.metrics.binary_f1 import BinaryMetric


class Accuracy(BinaryMetric):
    def get_metric(self):
        accuracy = accuracy_score(y_pred=self.predicts, y_true=self.goldens)
        return {
            'accuracy': accuracy,
            'fscore': 0
        }
    
    def report_all(self):
        raise NotImplementedError()