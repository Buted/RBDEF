import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, auc

from code.metrics.binary_f1 import BinaryMetric
from code.utils import JsonHandler


class ROCCurve:
    def __init__(self):
        self.metrics = [BinaryMetric() for _ in range(4)]
        self.metric_names = ["Routing", "Head", "Routing+Head", "Routing-double"]
    
    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, golden_labels, predict_labels):
        for p_l, metric in zip(predict_labels, self.metrics):
            metric.update(golden_labels, p_l)

    def report(self):
        return {'fscore': 0}

    def plot(self):
        data = []
        for metric, name in zip(self.metrics, self.metric_names):
            # metric.goldens = (1 - np.array(metric.goldens)).tolist()
            # metric.predicts = (1 - np.array(metric.predicts)).tolist()
            fpr, tpr, thredsholds = roc_curve(metric.goldens, metric.predicts)
            roc_auc = auc(fpr, tpr)
            data.append({
                "fpr": fpr.tolist(), 
                "tpr": tpr.tolist(),
                "auc": roc_auc
            })
            plt.plot(fpr, tpr, label="%s (area=%.2f)" % (name, roc_auc))
        JsonHandler.write_json('roc.json', data)
        plt.xlim([0.2, 0.5])
        plt.ylim([0.4, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig('ROC.png')