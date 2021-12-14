import os

from typing import Dict, List

from code.config import Hyper
from code.utils import JsonHandler


class GroupSplit:
    def __init__(self, source: str, hyper: Hyper):
        self.hyper = hyper
        self.data = self.read(source)

    def read(self, source: str) -> List[Dict]:
        filename = os.path.join(self.hyper.data_root, source)
        return JsonHandler.read_json(filename)
    
    def split(self) -> None:
        group2id = {}
        for data in self.data:
            role = data.get('entity').get('role')
            role_id = self.hyper.role2id[role]
            event = data.get('event_type')
            event_id = self.hyper.event2id[event]
            group = str(role_id) + '-' + str(event_id)
            if group not in group2id:
                group2id[group] = len(group2id) + 1
        self.write(group2id)

    def write(self, data):
        filename = os.path.join(self.hyper.data_root, "group2id.json")
        JsonHandler.write_json(filename, data)