import os
import random
import logging

from code.config import Hyper
from code.utils import JsonHandler


class FewshotDivider:
    def __init__(self, hyper: Hyper):
        # JsonHandler.merge_json([
        #     os.path.join(hyper.data_root, hyper.train),
        #     os.path.join(hyper.data_root, hyper.dev),
        #     os.path.join(hyper.data_root, hyper.test)
        # ], os.path.join(hyper.data_root, 'all.json'))
        self.data_root = hyper.data_root
        self.meta_roles = hyper.meta_roles
        self.data = JsonHandler.read_json(os.path.join(hyper.data_root, 'all.json'))
        self.role2id = hyper.role2id
        self.data = self._delete()
        
    def _delete(self):
        data = []
        for sample in self.data:
            if self.role2id[sample["entity"]["role"]] in self.meta_roles:
                data.append(sample)
        return data
    
    def generate_dataset(self):
        logging.info("Generate dataset")
        random.shuffle(self.data)
        trainset = self._generate_trainset()
        testset = self._generate_testset(trainset)
        JsonHandler.write_json(os.path.join(self.data_root, 'fewshot-train.json'), trainset)
        JsonHandler.write_json(os.path.join(self.data_root, 'fewshot-test.json'), testset)

    def _generate_trainset(self):
        trainset = []
        role2num = {role: 0 for role in self.meta_roles}
        for sample in self.data:
            role = self.role2id[sample["entity"]["role"]]
            if role2num[role] >= 5:
                continue
            trainset.append(sample)
            role2num[role] += 1
            if self._trainset_full(role2num):
                return trainset

    def _trainset_full(self, role2num):
        for num in role2num.values():
            if num < 5:
                return False
        return True

    def _generate_testset(self, trainset):
        testset = []
        for sample in self.data:
            if sample not in trainset:
                testset.append(sample)
        return testset