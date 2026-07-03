import pickle
from pathlib import Path
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models import Chunk
from .base import BaseRetriever


class TfidfRetriever(BaseRetriever):
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix: Any,
        corpus: list[dict[str, Any]],
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.corpus = corpus

    @classmethod
    def build(cls, chunks: list[Chunk], index_dir: str | Path) -> None:
        corpus = [chunk.model_dump() for chunk in chunks]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(chunk.text for chunk in chunks)
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "tfidf.pkl").open("wb") as file:
            pickle.dump((vectorizer, matrix, corpus), file)

    @classmethod
    def from_disk(cls, index_dir: str | Path) -> "TfidfRetriever":
        try:
            with (Path(index_dir) / "tfidf.pkl").open("rb") as file:
                vectorizer, matrix, corpus = pickle.load(file)
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(
                f"No index found at {index_dir!r}; run `index` first."
            ) from e
        return cls(vectorizer, matrix, corpus)

    def search(self, query: str, k: int) -> list[Chunk]:
        try:
            k = int(k)
        except (TypeError, ValueError) as e:
            raise ValueError(f"k must be an integer, got {k!r}") from e

        query = str(query)
        k = min(k, len(self.corpus))
        if k <= 0 or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().ravel()
        top = scores.argsort()[::-1][:k]
        return [Chunk(**self.corpus[i]) for i in top]
