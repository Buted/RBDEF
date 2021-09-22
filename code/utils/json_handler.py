import json

from typing import List


class JsonHandler:
    @staticmethod
    def read_json(filename: str) -> List[str]:
        with open(filename, 'r', encoding='utf-8') as reader:
            return json.load(reader)

    @staticmethod
    def write_json(filename: str, data: object) -> None:
        with open(filename, 'w', encoding='utf-8') as writter:
            json.dump(data, writter, indent=4)