import os

from collections import defaultdict
from typing import List, Dict

from code.utils import JsonHandler
from code.config import Hyper


class CoOccurStatistic:
    def __init__(self, hyper: Hyper):
        self.hyper = hyper
        self.entity_role_co_occur = defaultdict(lambda: [0] * len(hyper.id2role))
        self._build_co_occur_matrix()

    def _build_co_occur_matrix(self):
        for entity_type, role in self._yield_entity_and_role():
            role_id = self.hyper.role2id[role]
            self.entity_role_co_occur[entity_type][role_id] = 1

    def _yield_entity_and_role(self):
        for sample in self._read_data():
            yield sample["entity"]["entity_type"], sample["entity"]["role"]

    def _read_data(self) -> List[Dict]:
        filename = os.path.join(self.hyper.data_root, self.hyper.statistic)
        return JsonHandler.read_json(filename)

    def format_co_occur_matrix(self) -> str:
        formatted_matrix = "\n" + " "*13 + "".join("%-18s" % self.hyper.id2role[i] for i in range(self.hyper.role_vocab_size))
        line_format = lambda x: "".join("%-18d" % i for i in x)
        for i in range(len(self.entity_role_co_occur)):
            formatted_matrix += "\n"
            formatted_matrix += "%-13s" % self.hyper.id2entity[i]
            formatted_matrix += line_format(self.entity_role_co_occur[self.hyper.id2entity[i]])
        return formatted_matrix