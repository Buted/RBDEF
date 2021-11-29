import torch

from collections import defaultdict

import learn2learn as l2l

from code.config import Hyper
from code.dataloader.ace_dataloader import ACE_loader
from code.dataloader.meta_dataloader import FewShot_Dataset, Meta_Dataset
from code.dataloader.weighted_dataloader import FewRoleWithOther_Dataset, MetaWithOther_Dataset
from code.dataloader.few_role_dataloader import FewRole_Dataset
from code.dataloader.selector_dataloader import CoarseSelector_Dataset


class MetaFewRoleWithOther_Dataset(FewRoleWithOther_Dataset, Meta_Dataset):
    pass


class MetaFewRole_Dataset(FewRole_Dataset, Meta_Dataset):
    pass


class MetaCoarseSelector_Dataset(CoarseSelector_Dataset, Meta_Dataset):
    pass


class MetaMetaWithOther_Dataset(MetaWithOther_Dataset, Meta_Dataset):
    pass


class Balanced_loader:
    def __init__(self, dataset, hyper: Hyper) -> None:
        self.dataset = dataset
        self.hyper = hyper
        self.generator = self._get_balance_dataset()
        self.sampler = lambda dataset: iter(ACE_loader(
                dataset,
                batch_size=self.hyper.n*self.hyper.k,
                pin_memory=False,
                shuffle=True,
                num_workers=8,
            )).next()
        self.iter_times = 0
        self.iters_each_epoch = 250
        self.num_tasks = self.hyper.epoch_num * self.iters_each_epoch
    
    def _get_balance_dataset(self):
        remap = {i: i for i in range(len(self.hyper.meta_roles)+1)}
        dataset = l2l.data.MetaDataset(self.dataset)
        tasks = l2l.data.TaskDataset(dataset,
            task_transforms=[
                l2l.data.transforms.FusedNWaysKShots(dataset, n=self.hyper.n, k=self.hyper.k)
            ],
            num_tasks=self.num_tasks
        )
    
        for _ in range(self.num_tasks):
            task = tasks.sample()
            fewshot_dataset = FewShot_Dataset(dataset, task, remap)
            yield fewshot_dataset
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iter_times == self.iters_each_epoch:
            self.iter_times = 0
            raise StopIteration
        self.iter_times += 1
        dataset = next(self.generator)
        return self.sampler(dataset)

    def __len__(self):
        return self.iters_each_epoch


class BalancedWithOther_loader(Balanced_loader):
    def _get_balance_dataset(self):
        remap = defaultdict(int)
        for i, r in enumerate(self.hyper.meta_roles):
            remap[r] = i + 1
        
        dataset = l2l.data.MetaDataset(self.dataset)
        fewrole_tasks = l2l.data.TaskDataset(dataset,
            task_transforms=[
                l2l.data.transforms.FusedNWaysKShots(dataset, n=self.hyper.n-1, k=self.hyper.k, filter_labels=self.hyper.meta_roles)
            ],
            num_tasks=self.num_tasks
        )
        other_roles = list(set(range(self.hyper.role_vocab_size)) - set(self.hyper.meta_roles))
        other_tasks = l2l.data.TaskDataset(dataset,
            task_transforms=[
                l2l.data.transforms.FusedNWaysKShots(dataset, n=1, k=self.hyper.k, filter_labels=other_roles)
            ],
            num_tasks=self.num_tasks
        )

        for _ in range(self.num_tasks):
            fewrole_task, other_task = fewrole_tasks.sample(), other_tasks.sample()
            task = torch.cat((fewrole_task, other_task))
            fewshot_dataset = FewShot_Dataset(dataset, task, remap)
            yield fewshot_dataset
