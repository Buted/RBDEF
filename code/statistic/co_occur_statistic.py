import os

from collections import defaultdict
from typing import List, Dict

from code.utils import JsonHandler
from code.config import Hyper


class CoOccurStatistic:
    def __init__(self, hyper: Hyper):
        self.hyper = hyper
        self.entity_role_co_occur = defaultdict(lambda: [0] * len(hyper.id2role))
        self.event_role_co_occur = defaultdict(lambda: [0] * len(hyper.id2role))
        self._build_co_occur_matrix()

    def _build_co_occur_matrix(self):
        for entity_type, role, event_type in self._yield_mention_info():
            role_id = self.hyper.role2id[role]
            self.entity_role_co_occur[self.hyper.entity2id[entity_type]][role_id] = 1
            self.event_role_co_occur[self.hyper.event2id[event_type]][role_id] = 1

    def _yield_mention_info(self):
        for sample in self._read_data():
            yield sample["entity"]["entity_type"], sample["entity"]["role"], sample["event_type"]

    def _read_data(self) -> List[Dict]:
        filename = os.path.join(self.hyper.data_root, self.hyper.statistic)
        return JsonHandler.read_json(filename)
    
    def format_co_occur_matrix(self) -> str:
        output = "\nentity-role co-occur:" + self._format_matrix(self.entity_role_co_occur, self.hyper.id2entity)
        output += "\n\nevent-role co-occur:" + self._format_matrix(self.event_role_co_occur, self.hyper.id2event)
        return output

    def _format_matrix(self, co_occur_matrix: Dict[int, List[int]], id2key: Dict[int, str]) -> str:
        formatted_matrix = "\n" + " "*31 + "".join("%-18s" % self.hyper.id2role[i] for i in range(self.hyper.role_vocab_size))
        line_format = lambda x: "".join("%-18d" % i for i in x)
        for i in range(len(co_occur_matrix)):
            formatted_matrix += "\n"
            formatted_matrix += "%-31s" % id2key[i]
            formatted_matrix += line_format(co_occur_matrix[i])
        return formatted_matrix
    
    def save(self) -> None:
        filename = lambda x: os.path.join(self.hyper.data_root, x + '.json')
        self._save_matrix(self.entity_role_co_occur, filename('entity_role_co_occur'))
        self._save_matrix(self.event_role_co_occur, filename('event_role_co_occur'))        

    def _save_matrix(self, co_occur_matrix: Dict[int, List[int]], filename: str) -> None:
        matrix = {
            key_type: co_occur
            for key_type, co_occur in co_occur_matrix.items()
        }
        JsonHandler.write_json(filename, matrix)