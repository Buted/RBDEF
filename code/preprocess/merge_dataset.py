import os

from functools import partial

from code.utils import JsonHandler
from code.config import Hyper


def merge_dataset(hyper: Hyper) -> None:
    complete_path = partial(os.path.join, hyper.data_root)
    source = [complete_path(filename) for filename in hyper.raw_data_list]
    target = complete_path("all_data.json")

    JsonHandler.merge_json(source, target)