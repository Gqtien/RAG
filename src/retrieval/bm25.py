import bm25s
from pathlib import Path
from src.models import Chunk
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, bm25: bm25s.BM25) -> None:
        self.bm25 = bm25

    @classmethod
    def from_disk(cls, index_dir: str | Path) -> "BM25Retriever":
        try:
            bm25 = bm25s.BM25.load(index_dir, load_corpus=True)
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(
                f"No index found at {index_dir!r}; run `index` first."
            ) from e
        return cls(bm25)

    def search(self, query: str, k: int) -> list[Chunk]:
        try:
            k = int(k)
        except (TypeError, ValueError) as e:
            raise ValueError(f"k must be an integer, got {k!r}") from e

        query = str(query)
        k = min(k, int(self.bm25.scores["num_docs"]))
        if k <= 0 or not query.strip():
            return []

        tokens = bm25s.tokenize(query, show_progress=False)
        results, _ = self.bm25.retrieve(tokens, k=k, show_progress=False)
        return [Chunk(**results[0, i]) for i in range(results.shape[1])]
