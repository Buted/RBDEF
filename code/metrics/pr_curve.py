import logging

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

from code.metrics.binary_f1 import BinaryMetric


class BinaryPRCurve(BinaryMetric):
    def __init__(self):
        super(BinaryPRCurve, self).__init__()
        
    def get_metric(self, plot: bool = False):
        p, r, threshold = precision_recall_curve(self.goldens, self.predicts)
        
        format = lambda name, values: "%s: " % name + ",".join(map(str, values.tolist()))
        logging.info(format("p", p))
        logging.info(format("r", r))
        logging.info(format("threshold", threshold))

        if plot:
            plt.figure("P-R Curve")
            plt.title("Precision/Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.plot(r, p)
            plt.show()
            plt.savefig("P-R Curve.png")

        max_threshold = 0.
        max_f1 = 0.
        for cur_p, cur_r, thre in zip(p, r, threshold):
            f1 = 2*cur_p*cur_r / (cur_p + cur_r)
            if f1 >= max_f1:
                max_f1 = f1
                max_threshold = thre
        logging.info("max_threshold: %f, max_f1: %f" % (max_threshold, max_f1))
        print("max_threshold: %f, max_f1: %f" % (max_threshold, max_f1))