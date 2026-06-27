from typing import Protocol
from src.models import Chunk


class BaseRetriever(Protocol):
    def search(self, query: str, k: int) -> list[Chunk]: ...
