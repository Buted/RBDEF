
from typing import List, Dict

from code.dataloader.ace_dataloader import ACE_Dataset
from code.config import Hyper


class FewRole_Dataset(ACE_Dataset):
    def __init__(self, hyper: Hyper, dataset: str, select_roles: List[int]):
        super(FewRole_Dataset, self).__init__(hyper, dataset)
        # self._remap_labels(select_roles)
        self._delete_unuseless_roles(select_roles)
        
    def _delete_unuseless_roles(self, select_roles: List[int]):
        using_sample_ids = [i for i in range(len(self.label)) if self.label[i] in select_roles]
        self._delete_fields(using_sample_ids)
        
        self._remap_labels(select_roles)

    def _delete_fields(self, using_sample_ids: List[int]):
        sample_cnt = len(self.label)

        delete_field = lambda field: [field[i] for i in range(sample_cnt) if i in using_sample_ids]
        self.tokens = delete_field(self.tokens)
        self.label = delete_field(self.label)
        self.entity_start  = delete_field(self.entity_start)
        self.entity_end = delete_field(self.entity_end)
        self.entity_type = delete_field(self.entity_type)
        self.entity_id = delete_field(self.entity_id)
        self.event_id = delete_field(self.event_id)
        self.trigger_start = delete_field(self.trigger_start)
        self.trigger_end = delete_field(self.trigger_end)

    def _remap_labels(self, select_roles: List[int]):
        role_remap = self._build_role_remap(select_roles)
        
        self.label = [role_remap[r] for r in self.label]

    def _build_role_remap(self, select_roles: List[int]):
        role_remap = {r: i for i, r in enumerate(select_roles)}
        return role_remap


class HeadRole_Dataset(FewRole_Dataset):
    def __init__(self, hyper: Hyper, dataset: str, select_roles: List[int]):
        reverse_roles = list(range(hyper.role_vocab_size))
        for role in select_roles:
            reverse_roles.remove(role)
        reverse_roles.remove(0)
        super(HeadRole_Dataset, self).__init__(hyper, dataset, reverse_roles)


class Branch_Dataset(FewRole_Dataset):
    def __init__(self, hyper: Hyper, dataset: str, select_roles: List[int]):
        reverse_roles = list(range(1, hyper.role_vocab_size))
        super(Branch_Dataset, self).__init__(hyper, dataset, reverse_roles)
        self.select_roles = select_roles
        self.label = [self._relabel(role) for role in self.label]
    
    def _relabel(self, role: int) -> int:
        return 0 if role not in self.select_roles else 1