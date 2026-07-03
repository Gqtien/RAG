from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def covers(
    retrieved: MinimalSource,
    correct: MinimalSource,
    max_context_length: int,
    threshold: float,
) -> bool:
    if retrieved.file_path != correct.file_path:
        return False

    retrieved_last = min(
        retrieved.last_character_index,
        retrieved.first_character_index + max_context_length,
    )
    overlap = min(retrieved_last, correct.last_character_index) - max(
        retrieved.first_character_index, correct.first_character_index
    )
    correct_len = correct.last_character_index - correct.first_character_index
    return correct_len > 0 and overlap / correct_len >= threshold


def recall_at_k(
    retrieved: list[MinimalSource],
    correct: list[MinimalSource],
    k: int,
    max_context_length: int,
    threshold: float,
) -> float:
    top_k = retrieved[:k]
    found = sum(
        any(covers(r, c, max_context_length, threshold) for r in top_k)
        for c in correct
    )
    return found / len(correct)


def evaluate(
    results: StudentSearchResults,
    dataset: RagDataset,
    ks: list[int],
    max_context_length: int,
    threshold: float,
) -> dict[int, float]:
    truth = {
        question.question_id: question
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion) and question.sources
    }
    totals = {k: 0.0 for k in ks}
    evaluated = 0
    for result in results.search_results:
        question = truth.get(result.question_id)
        if question is None:
            continue
        evaluated += 1
        for k in ks:
            totals[k] += recall_at_k(
                result.retrieved_sources,
                question.sources,
                k,
                max_context_length,
                threshold,
            )
    return {k: (totals[k] / evaluated if evaluated else 0.0) for k in ks}
