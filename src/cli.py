from functools import lru_cache
from pathlib import Path
from typing import Any
from tqdm import tqdm
from src.config import load_config
from src.evaluation import evaluate as evaluate_recall
from src.ingestion import chunk_files, load_files
from src.retrieval import build_index, make_retriever
from src.generation import build_prompt, generate
from src.models import (
    Chunk,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class RagCLI:
    def __init__(self, config: str = "config.yml") -> None:
        self.config = load_config(config)

    def index(self, max_chunk_size: int | None = None) -> None:
        if max_chunk_size is not None:
            self.config.indexing.max_chunk_size = max_chunk_size

        files = load_files(self.config.paths.raw_dir)
        chunks = chunk_files(self.config.indexing, files)
        build_index(self.config, chunks)
        print(
            f"Ingestion complete! {len(chunks)} chunks from {len(files)} "
            f"files saved under {self.config.paths.index_dir!r}"
        )

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        retriever = make_retriever(self.config)
        hits = retriever.search(query, k)
        return [source.model_dump() for source in to_sources(hits)]

    def search_dataset(
        self,
        dataset_path: str | Path,
        save_directory: str | Path,
        k: int = 10,
    ) -> None:
        dataset = RagDataset.model_validate_json(read_file(dataset_path))
        retriever = make_retriever(self.config)

        results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            hits = retriever.search(question.question, k)
            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=to_sources(hits),
                )
            )
        output = StudentSearchResults(search_results=results, k=k)

        save_path = Path(save_directory) / Path(dataset_path).name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Saved student_search_results to {save_path}")

    def answer(self, query: str, k: int = 10) -> None:
        retriever = make_retriever(self.config)
        chunks = retriever.search(query, k)
        prompt = build_prompt(
            query,
            chunks,
            self.config.model.max_context_length,
        )
        print(generate(self.config.model, prompt))

    def answer_dataset(
        self,
        student_search_results_path: str | Path,
        save_directory: str | Path,
    ) -> None:
        results = StudentSearchResults.model_validate_json(
            read_file(student_search_results_path)
        )
        total = len(results.search_results)
        print(f"Loaded {total} questions from {student_search_results_path}")

        raw_dir = self.config.paths.raw_dir
        max_context_length = self.config.model.max_context_length
        answers: list[MinimalAnswer] = []
        for result in tqdm(results.search_results, desc="Answering"):
            chunks = sources_to_chunks(raw_dir, result.retrieved_sources)
            prompt = build_prompt(result.question, chunks, max_context_length)
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=generate(self.config.model, prompt),
                )
            )
        print(f"Processed {len(answers)} of {total} questions")

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=results.k
        )
        name = Path(student_search_results_path).name
        save_path = Path(save_directory) / name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Saved student_search_results_and_answer to {save_path}")

    def evaluate(
        self,
        student_search_results_path: str | Path,
        dataset_path: str | Path,
        k: int = 10,
    ) -> None:
        results = StudentSearchResults.model_validate_json(
            read_file(student_search_results_path)
        )
        dataset = RagDataset.model_validate_json(read_file(dataset_path))
        ks = [n for n in self.config.evaluation.recall_k if n <= k]
        scores = evaluate_recall(
            results,
            dataset,
            ks,
            self.config.model.max_context_length,
            self.config.evaluation.overlap_threshold,
        )

        print("Evaluation Results")
        print("=" * 15)
        print(f"Questions evaluated: {len(results.search_results)}")
        for n in ks:
            print(f"Recall@{n}: {scores[n]:.3f}")


def read_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"{str(path)!r} is not a readable file.")
    return file_path.read_text(encoding="utf-8")


def to_sources(chunks: list[Chunk]) -> list[MinimalSource]:
    return [
        MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )
        for chunk in chunks
    ]


@lru_cache(maxsize=None)
def read_source_text(raw_dir: str, file_path: str) -> str | None:
    try:
        return (Path(raw_dir) / file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def sources_to_chunks(
    raw_dir: str, sources: list[MinimalSource]
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source in sources:
        text = read_source_text(raw_dir, source.file_path)
        if text is None:
            continue
        chunks.append(
            Chunk(
                file_path=source.file_path,
                first_character_index=source.first_character_index,
                last_character_index=source.last_character_index,
                text=text[
                    source.first_character_index:source.last_character_index
                ],
            )
        )
    return chunks
