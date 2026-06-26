import bm25s
from src.models import Chunk


def index_chunks(chunks: list[Chunk], index_dir: str) -> None:
    texts = [chunk.text for chunk in chunks]
    corpus = [chunk.model_dump() for chunk in chunks]
    tokens = bm25s.tokenize(texts)
    bm25 = bm25s.BM25(corpus=corpus)
    bm25.index(tokens)
    bm25.save(index_dir, corpus=corpus)
