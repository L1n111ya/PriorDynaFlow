import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Iterable, Any


class Reader(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """ To be overriden by the descendant class """


class JSONLReader(Reader):
    def parse_file(file_path: Path) -> list:
        with open(file_path, "r", encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
            # text = '\n'.join([str(line) for line in lines])
        return lines  # text

    def parse(file_path: Path) -> str:
        with open(file_path, "r", encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
            text = '\n'.join([str(line) for line in lines])
        return text


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]