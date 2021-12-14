import os

from collections import defaultdict
from typing import Dict, List

from code.config import Hyper
from code.utils import JsonHandler


class RoleNumberCounter:
    def __init__(self, source: str, hyper: Hyper):
        self.hyper = hyper
        self.data = self.read(source)

    def read(self, source: str) -> List[Dict]:
        filename = os.path.join(self.hyper.data_root, source)
        return JsonHandler.read_json(filename)
    
    def count(self) -> None:
        role2num = defaultdict(int)
        for data in self.data:
            role = data.get('entity').get('role')
            role_id = self.hyper.role2id[role]
            role2num[role_id] += 1
        self.write(role2num)

    def write(self, data):
        filename = os.path.join(self.hyper.data_root, "role2num.json")
        JsonHandler.write_json(filename, data)