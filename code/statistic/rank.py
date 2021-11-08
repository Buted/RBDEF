import re
import logging

import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict


class Ranker(object):
    def __init__(self, role_vocab_size: int, role_ids: List[int]=None) -> None:
        self.role_vocab_size = role_vocab_size
        self.role_ids = role_ids or list(range(self.role_vocab_size))
        self._init_values()
        self._init_matchers()

    def _init_values(self) -> None:
        defaultlist = lambda: [[] for _ in range(self.role_vocab_size)]
        self.values = {
            'pre': defaultlist(), 
            'rec': defaultlist(),
            'f1': defaultlist(),
            'golden': defaultlist(),
            'unrelated': defaultlist(),
            'indicator': defaultlist()
        }

    def _init_matchers(self) -> None:
        self.f1_matcher = \
            r'^(\d+).*precision: (\d\.\d{4}), recall: (\d\.\d{4}), fscore: (\d\.\d{4})'
        self.indicator_matcher = \
            r'^(\d+).*golden: (\d\.\d+), unrelated: (\d+\.\d+), indicator: (\d\.\d+)'
    
    def match_file(self, filename: str) -> None:
        for line in self.read_file(filename):
            line = line.strip()
            self._match_one_line(line)
            if self._stop_match():
                break
        self._transform_values_to_np()

    @staticmethod
    def read_file(filename: str) -> List[str]:
        with open(filename, 'r', encoding="utf-8") as reader:
            return reader.readlines()

    def _match_one_line(self, line: str) -> None:
        self._match_f1(line)
        self._match_indicator(line)

    def _match_f1(self, line: str):
        m = re.match(self.f1_matcher, line)
        if m:
            role_idx = int(m.group(1))
            role_idx = role_idx if role_idx < self.role_vocab_size else 0
            self.values['pre'][role_idx].append(float(m.group(2)))
            self.values['rec'][role_idx].append(float(m.group(3)))
            self.values['f1'][role_idx].append(float(m.group(4)))

    def _match_indicator(self, line: str):
        m = re.match(self.indicator_matcher, line)
        if m:
            role_idx = int(m.group(1))
            self.values['golden'][role_idx].append(float(m.group(2)))
            self.values['unrelated'][role_idx].append(1 - float(m.group(3)))
            self.values['indicator'][role_idx].append(float(m.group(4)))

    def _stop_match(self) -> bool:
        return len(self.values['golden'][-1]) >= 30
    
    def _transform_values_to_np(self) -> None:
        np_values = {value_name: np.array(values) for value_name, values in self.values.items()}
        self.values = np_values
        
    def ranking(self):
        self.rank = {epoch: self._rank_in_epoch(epoch) 
            for epoch in range(self.values['pre'].shape[-1])}

    def _rank_in_epoch(self, epoch: int) -> Dict[str, List[int]]:
        rank = {
            value_name: self._rank_list(values[:, epoch][self.role_ids])
            for value_name, values in self.values.items()
        }

        indicators = [(rank['golden'][i] + rank['unrelated'][i]) / 2 
            for i in range(len(self.role_ids))]
        # indicators = [rank['unrelated'][i] for i in range(len(self.role_ids))]
        indicators = self._rank_list(indicators)
        indicators = [(f1 + 0.5 * indicator) / 2 for f1, indicator in zip(rank['f1'], indicators)]
        rank['indicator'] = self._rank_list(indicators)

        return rank

    def _rank_list(self, values: List[float]):
        rank_values = pd.Series(values)
        rank_values = rank_values.rank(method='min')
        return [rank_values[i] for i in range(len(values))]

    def save_as_img(self, filename: str) -> None:
        plot_epochs = [0, 1, 2]
        subfig_num = len(plot_epochs)
        fig = plt.figure(figsize=(30, 10))
        for i, epoch in enumerate(plot_epochs):
            self._plot_subfig(fig, subfig_num, i, epoch)
        plt.xlabel("Role")
        plt.ylabel("Rank")
        plt.savefig(filename)

    def _plot_subfig(self, fig, subfig_num: int, subfig_idx: int, epoch: int):
        ax = fig.add_subplot(1, subfig_num, subfig_idx+1)
        ax.set_title("Epoch %d" % epoch)
        x_axix = list(range(len(self.role_ids)))
        for value_name, values in self.rank[epoch].items():
            ax.plot(x_axix, values, label=str(value_name))
        ax.legend()
    
    def save_into_log(self) -> None:
        score_log = lambda name, score: "%-10s: %s" % (name, 
            ", ".join("%-2d" % s for s in score)
        )
        for epoch, rank in self.rank.items():
            log = "Epoch %d" % epoch
            log += "\n" + "%-10s: " % "Role id"
            log += " ".join("%-3d" % i for i in range(self.role_vocab_size))
            for rank_metric, score in rank.items():
                log += "\n"
                log += score_log(rank_metric, score)
            logging.info(log)
