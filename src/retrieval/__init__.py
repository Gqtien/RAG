from src.models import Chunk, Config

from .base import BaseRetriever
from .bm25 import BM25Retriever
from .tfidf import TfidfRetriever

RETRIEVERS: dict[str, type[BaseRetriever]] = {
    "bm25": BM25Retriever,
    "tfidf": TfidfRetriever,
}


def retriever_cls(config: Config) -> type[BaseRetriever]:
    method = config.indexing.retrieval_method
    if method not in RETRIEVERS:
        raise ValueError(f"Unknown retrieval method: {method!r}")
    return RETRIEVERS[method]


def build_index(config: Config, chunks: list[Chunk]) -> None:
    retriever_cls(config).build(chunks, config.paths.index_dir)


def make_retriever(config: Config) -> BaseRetriever:
    return retriever_cls(config).from_disk(config.paths.index_dir)


__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "TfidfRetriever",
    "build_index",
    "make_retriever",
]
