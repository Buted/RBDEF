import os
import json
import logging

from typing import List, Dict, Tuple

from code.config import NonRole, NonEntity, NonEvent
from code.utils import JsonHandler


class Entity:
    def __init__(self, entity_start: int, entity_end: int, entity_type: str):
        self.start = entity_start
        self.end = entity_end
        self.entity_type = entity_type

    def __eq__(self, other) -> bool:
        return self.start == other.start and self.end == other.end and self.entity_type == other.entity_type


class Role:
    def __init__(self, entity: Entity, role: str):
        self.entity = entity
        self.role = role
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Entity):
            return self.has_same_entity(other)
        elif isinstance(other, Role):
            if other.entity == self.entity:
                logging.info("The entity (type: %s, start: %d) plays multi-roles in an event!" % (self.entity_type, self.start))
                return True
            return False     
        else:
            raise NotImplementedError("Use has_same_entity instead.")

    def has_same_entity(self, entity: Entity) -> bool:
        return self.entity == entity

    @property
    def start(self) -> int:
        return self.entity.start
    
    @property
    def end(self) -> int:
        return self.entity.end
    
    @property
    def entity_type(self) -> str:
        return self.entity.entity_type

    def __hash__(self) -> int:
        return hash(str(self.start) + self.role + str(self.end))

class Sentence:
    def __init__(self, sentence: Dict):
        self.sentence = sentence
        self.entities = self._extract_entities()
        self.event2arguments = {}
        self._extract_events()

    def _extract_entities(self) -> List[Entity]:
        return [Sentence._extract_entity(e_info) for e_info in self.sentence["golden-entity-mentions"]]

    @staticmethod
    def _extract_entity(entity_info: Dict) -> Entity:
        return Entity(entity_info["start"], entity_info["end"], entity_info["entity-type"])

    def _extract_events(self):
        for event_info in self.sentence["golden-event-mentions"]:
            event = (event_info["event_type"], event_info["trigger"]["text"])
            arguments = list(set(Role(self._extract_entity(argument), argument["role"]) for argument in event_info["arguments"]))
            self.event2arguments[event] = (arguments, event_info["trigger"])

    def get_entity_types(self) -> List[str]:
        return [ent.entity_type for ent in self.entities]
    
    def get_event_types(self) -> List[str]:
        return [event[0] for event in self.event2arguments]

    def get_role_types(self) -> List[str]:
        return [argument.role for event in self.event2arguments for argument in self.event2arguments[event][0]]

    def gen_samples(self) -> List[Dict]:
        if not self.entities:
            return []
        
        samples = []
        for entity in self.entities:
            for event in self.event2arguments:
                is_role = False
                for role in self.event2arguments[event][0]:
                    if role == entity:
                        is_role = True
                        samples.append(self._gen_one_sample(role, event[0], self.event2arguments[event][1]))
                if not is_role:
                    samples.append(self._gen_non_role_sample(entity, event[0], self.event2arguments[event][1]))
        return samples            

    def _gen_one_sample(self, entity: Role, event_type: str, trigger: Dict) -> Dict:
        return {
            "sentence": self.sentence["sentence"],
            "words": self.sentence["words"],
            "entity": {
                "start": entity.start,
                "end": entity.end,
                "entity_type": entity.entity_type,
                "role": entity.role
            },
            "event_type": event_type,
            "trigger": trigger
        }
    
    def _gen_non_role_sample(self, entity: Entity, event_type: str, trigger: Dict) -> Dict:
        return {
            "sentence": self.sentence["sentence"],
            "words": self.sentence["words"],
            "entity": {
                "start": entity.start,
                "end": entity.end,
                "entity_type": entity.entity_type,
                "role": NonRole
            },
            "event_type": event_type,
            "trigger": trigger
        }
        

class ACE_Preprocessor:
    def __init__(self, hyper):
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root
        self.raw_data_list = hyper.raw_data_list

        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        self.entity2id = {NonEntity: 0}
        self.event2id = {NonEvent: 0}
        self.role2id = {NonRole: 0}
        
    def gen_type_vocab(self) -> None:
        logging.info("Write entity2id, event2id and role2id vocabularies.")
        for path in self.raw_data_list:
            if "train" not in path:
                continue
            source = os.path.join(self.raw_data_root, path)
            entity_target = os.path.join(self.data_root, 'entity2id.json')
            event_target = os.path.join(self.data_root, 'event2id.json')
            role_target = os.path.join(self.data_root, 'role2id.json')
            
            for sentence in JsonHandler.read_json(source):
                self._complete_entity_and_event_vocab(sentence)
            
            JsonHandler.write_json(entity_target, self.entity2id)
            JsonHandler.write_json(event_target, self.event2id)
            JsonHandler.write_json(role_target, self.role2id)
        
    def _complete_entity_and_event_vocab(self, sentence: Dict) -> None:
        sen = Sentence(sentence)
        self._build_type_vocab(self.entity2id, sen.get_entity_types())
        self._build_type_vocab(self.event2id, sen.get_event_types())
        self._build_type_vocab(self.role2id, sen.get_role_types())
            
    @staticmethod
    def _build_type_vocab(vocab: Dict, occurred_types: List[str]) -> None:
        for ent_type in occurred_types:
            if ent_type not in vocab:
                vocab[ent_type] = len(vocab)

    def gen_all_data(self) -> None:
        for path in self.raw_data_list:
            self._gen_one_data(path)

    def _gen_one_data(self, dataset: str) -> None:
        logging.info("Preprocess %s" % dataset.split('.')[0])
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)

        samples = []
        for sentence in JsonHandler.read_json(source):
            samples.extend(self._process_sentence(sentence))
        
        JsonHandler.write_json(target, samples)

    def _process_sentence(self, sentence: Dict) -> List[Dict]:
        sentence = Sentence(sentence)
        return sentence.gen_samples()