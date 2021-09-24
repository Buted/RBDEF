import os

from code.utils import JsonHandler
from code.config import Hyper


def merge_dataset(hyper: Hyper) -> None:
    source = [os.path.join(hyper.raw_data_root, filename) for filename in hyper.raw_data_list]
    target = os.path.join(hyper.data_root, "all_data.json")

    JsonHandler.merge_json(source, target)