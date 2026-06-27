from src.models import Config
from .base import BaseRetriever
from .bm25 import BM25Retriever


def make_retriever(config: Config) -> BaseRetriever:
    method = config.indexing.retrieval_method
    match method:
        case "bm25":
            return BM25Retriever.from_disk(config.paths.index_dir)

    raise ValueError(f"Unknown retrieval_method: {method!r}")


__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "make_retriever",
]
