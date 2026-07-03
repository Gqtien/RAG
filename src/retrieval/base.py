from abc import ABC, abstractmethod
from pathlib import Path
from src.models import Chunk


class BaseRetriever(ABC):

    @classmethod
    @abstractmethod
    def build(cls, chunks: list[Chunk], index_dir: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def from_disk(cls, index_dir: str | Path) -> "BaseRetriever": ...

    @abstractmethod
    def search(self, query: str, k: int) -> list[Chunk]: ...
