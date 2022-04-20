import re

import matplotlib.pyplot as plt

from typing import List


class IndicatorMatcher(object):
    def __init__(self, role_ids: List[int]) -> None:
        self.role_ids = role_ids
        self._init_values()
        self._init_matchers()

    def _init_values(self) -> None:
        defaultvalues = lambda: {
            'pre': list(), 
            'rec': list(),
            'f1': list(),
            'golden': list(),
            'unrelated': list(),
            'indicator': list()
        }
        self.role2values = {role_i: defaultvalues() for role_i in self.role_ids}

    def _init_matchers(self) -> None:
        f1_matcher = lambda i: \
            r'^%d.*precision: (\d\.\d{4}), recall: (\d\.\d{4}), fscore: (\d\.\d{4})' % i
        indicator_matcher = lambda i: \
            r'^%d.*golden: (\d\.\d{4}), unrelated: (\d\.\d{4}), indicator: (\d\.\d{4})' % i
        self.role2f1_matchers = {role_i: f1_matcher(role_i) for role_i in self.role_ids}
        self.role2indicator_matchers = {role_i: indicator_matcher(role_i) for role_i in self.role_ids}
    
    def match_file(self, filename: str) -> None:
        for line in self.read_file(filename):
            line = line.strip()
            self._match_one_line(line)
            if self._stop_match():
                break

    @staticmethod
    def read_file(filename: str) -> List[str]:
        with open(filename, 'r', encoding="utf-8") as reader:
            return reader.readlines()

    def _match_one_line(self, line: str) -> None:
        m = None
        for role_id, rstr in self.role2f1_matchers.items():
            m = re.match(rstr, line)
            if m:
                self.role2values[role_id]['pre'].append(float(m.group(1)))
                self.role2values[role_id]['rec'].append(float(m.group(2)))
                self.role2values[role_id]['f1'].append(float(m.group(3)))
                break
        for role_id, rstr in self.role2indicator_matchers.items():
            m = re.match(rstr, line)
            if m:
                self.role2values[role_id]['golden'].append(float(m.group(1)))
                self.role2values[role_id]['unrelated'].append(1 - float(m.group(2)))
                self.role2values[role_id]['indicator'].append(float(m.group(3))/2)
                break
    def _stop_match(self) -> bool:
        return len(self.role2values[max(self.role_ids)]['golden']) >= 30

    def save_as_img(self, filename: str) -> None:
        fig = plt.figure(figsize=(30, 10))
        for i, role_id in enumerate(self.role_ids):
            ax = fig.add_subplot(1, len(self.role_ids), i+1)
            ax.set_title("Role %d" % role_id)
            x_axix = list(range(len(list(self.role2values[role_id].values())[0])))
            for value_name, values in self.role2values[role_id].items():
                ax.plot(x_axix, values, label=str(value_name))
            ax.legend()
        plt.xlabel("Epoch")
        plt.savefig(filename)