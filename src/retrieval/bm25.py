from pathlib import Path
import bm25s
from src.models import Chunk
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, bm25: bm25s.BM25) -> None:
        self.bm25 = bm25

    @classmethod
    def build(cls, chunks: list[Chunk], index_dir: str | Path) -> None:
        texts = [chunk.text for chunk in chunks]
        corpus = [chunk.model_dump() for chunk in chunks]
        bm25 = bm25s.BM25(corpus=corpus)
        bm25.index(bm25s.tokenize(texts))
        bm25.save(index_dir, corpus=corpus)

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
