import os

from typing import List, Tuple

from code.config import Hyper
from code.utils import JsonHandler


class MetaInfo(object):
    def __init__(self, hyper: Hyper, dataset: str):
        self.meta_roles = hyper.meta_roles
        self.meta_entity, self.meta_event = self._extract_meta_entity_and_event(hyper, dataset)
    
    def _extract_meta_entity_and_event(self, hyper: Hyper, dataset: str) -> Tuple[List[int]]:
        meta_entity = set()
        meta_event = set()
        for sample in JsonHandler.read_json(os.path.join(hyper.data_root, dataset)):
            entity = sample["entity"]
            label = hyper.role2id[entity["role"]]
            if label not in self.meta_roles:
                continue

            entity_type = hyper.entity2id[entity["entity_type"]]
            meta_entity.add(entity_type)

            event_type = hyper.event2id[sample["event_type"]]
            meta_event.add(event_type)
        return list(meta_entity), list(meta_event)
    
    def save(self, hyper: Hyper):
        filename = os.path.join(hyper.data_root, 'meta_info.json')
        info = {
            "meta_roles": self.meta_roles,
            "meta_entity": self.meta_entity,
            "meta_event": self.meta_event
        }
        JsonHandler.write_json(filename, info)