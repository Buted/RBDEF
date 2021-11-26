from code.config import Hyper
from code.metrics import F1
from code.metrics.binary_f1 import BinaryMetric


class MultiTaskF1:
    def __init__(self, hyper: Hyper):
        self.main_f1 = F1(hyper)
        self.binary_f1 = BinaryMetric()
    
    def reset(self):
        self.main_f1.reset()
        self.binary_f1.reset()
    
    def update(self, golden_labels, predict_labels) -> None:
        main_goldens, binary_goldens = golden_labels
        main_predicts, binary_predicts = predict_labels

        self.main_f1.update(main_goldens, main_predicts)
        self.binary_f1.update(binary_goldens, binary_predicts)
    
    def report(self):
        main_report = self.main_f1.report()
        binary_report = self.binary_f1.get_metric()
        report = {'fscore': 0}
        for key, values in main_report.items():
            report['main_'+key] = values
        
        for key, values in binary_report.items():
            report['binary_'+key] = values

        return report
    
    def report_all(self):
        main_report = self.main_f1.report_all()
        binary_report = self.binary_f1.report_all()
        return main_report + binary_report