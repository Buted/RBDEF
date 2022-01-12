import os
import random
import logging

from collections import defaultdict
from typing import Dict, List, Tuple

from code.utils import JsonHandler
from code.config import Hyper


class DatasetDivider:
    def __init__(self, hyper: Hyper):
        self.hyper = hyper
        self.divided_data_root = hyper.divided_data_root
        if not os.path.exists(self.divided_data_root):
            os.makedirs(self.divided_data_root)

        read = lambda x: JsonHandler.read_json(os.path.join(self.hyper.data_root, x))
        self.testset = read(hyper.test)
        self.devset = read(hyper.dev)
        self.trainset = read(hyper.train)
        self._merge_samples_belong_to_same_sentence()

    def divide(self, k: int=5):
        self._divide(k)
        self._write()
        self._print_distributions()

    def _merge_samples_belong_to_same_sentence(self):
        self.sentence2samples = defaultdict(list)
        self.role2sentences = defaultdict(set)
        self._merge(self.testset)
        self._merge(self.devset)
        self._merge(self.trainset)
        self.sentence_ids = list(self.sentence2samples.keys())

    def _merge(self, dataset: List[Dict]):
        for sample in dataset:
            sentence_id = hash(sample["sentence"])
            self.sentence2samples[sentence_id].append(sample)

            role = self._extract_role(sample)
            self.role2sentences[role].add(sentence_id)

    def _write(self):
        write = lambda file, data: JsonHandler.write_json(os.path.join(self.divided_data_root, file), data)

        write('test_all.json', self.testset_all)
        write(self.hyper.test, self.testset)
        write(self.hyper.dev, self.devset)
        write(self.hyper.train, self.trainset)
        
    def _divide(self, k: int):
        testset_ids = self._extract_testset(k)
        self._divide_devset_and_trainset(testset_ids, k)

    def _extract_testset(self, k: int) -> List[int]:
        sorted_role2sentences = sorted(self.role2sentences.items(), key=lambda x: (len(x[1]), x[0]))
        self.testset = []
        self.testset_all = []
        testset_ids = []
        role2num = defaultdict(int)
        for role, sentence_ids in sorted_role2sentences:
            if role2num[role] >= k:
                continue

            self._split_samples_from_sentence(k, testset_ids, role2num, sentence_ids)

        random.shuffle(self.sentence_ids)
        for s_id in self.sentence_ids:
            if len(self.testset_all) > 3100:
                break

            if s_id in testset_ids:
                continue

            self._append_to_testset(k, testset_ids, role2num, s_id)

        return testset_ids

    def _split_samples_from_sentence(self, k: int, testset_ids: List[int], role2num: Dict[int, int], sentence_ids: List[int]):
        sentence_ids = list(sentence_ids)
        random.shuffle(sentence_ids)

        for sen_id in sentence_ids[:k]:
            if sen_id in testset_ids:
                continue
                
            self._append_to_testset(k, testset_ids, role2num, sen_id)

    def _append_to_testset(self, k: int, testset_ids: List[int], role2num: Dict[int, int], sen_id: int):
        testset_ids.append(sen_id)
        samples = self.sentence2samples[sen_id]
        self._extract_samples(k, role2num, samples)

    def _extract_samples(self, k: int, role2num: Dict[int, int], samples: List[Dict]):
        for samp in samples:
            self.testset_all.append(samp)
            role = self._extract_role(samp)
                    
            if role2num[role] >= k:
                continue
                    
            role2num[role] += 1
            self.testset.append(samp)

    def _extract_role(self, sample: Dict):
        entity = sample["entity"]
        role = self.hyper.role2id[entity["role"]]
        return role

    def _divide_devset_and_trainset(self, testset_ids: List[int], k: int):
        # sentence_num = len(self.sentence_ids)
        # no_testset_num = sentence_num - len(testset_ids)
        # devset_num = no_testset_num // 8
        self.devset, self.trainset = [], []
        for i in self.sentence_ids:
            if i in testset_ids:
                continue
            
            # if 20 in [self._extract_role(sample) for sample in self.sentence2samples[i]]:
                # belong_to_dev = False
            # else:
                # belong_to_dev = True if random.randint(1, 100) <= 13 else False
            belong_to_dev = True if random.randint(1, 100) <= 13 else False
            if belong_to_dev:
                self._add_samples(self.devset, self.sentence2samples[i])
            else:
                self._add_samples(self.trainset, self.sentence2samples[i])
    
    def _add_samples(self, dataset: List, samples: List):
        dataset.extend(samples)

    def _print_distributions(self):
        logging.info("Testset (normal):")
        self._print_dataset_distribution(self.testset_all)
        logging.info("Testset (balanced):")
        self._print_dataset_distribution(self.testset)
        logging.info("Devset:")
        self._print_dataset_distribution(self.devset)
        logging.info("Trainset:")
        self._print_dataset_distribution(self.trainset)

    def _print_dataset_distribution(self, dataset: List[Dict]):
        logging.info("Samples: %d" % len(dataset))
        role2num = defaultdict(int)
        for sample in dataset:
            role = self._extract_role(sample)
            role2num[role] += 1
        
        for i in range(len(self.role2sentences)):
            logging.info("Role: %d, Samples: %d" % (i, role2num[i]))