from src.models import Chunk

SYSTEM = (
    "Answer the question using only the context below. "
    "If the answer is not in the context, say you don't know."
)


def format_context(chunks: list[Chunk], max_context_length: int) -> str:
    parts: list[str] = []
    budget = max_context_length
    for chunk in chunks:
        if budget <= 0:
            break
        snippet = chunk.text[:budget]
        parts.append(snippet)
        budget -= len(snippet)
    return "\n\n".join(parts)


def build_prompt(
    question: str,
    chunks: list[Chunk],
    max_context_length: int,
) -> str:
    context = format_context(chunks, max_context_length)
    return f"{SYSTEM}\n\nContext:\n{context}\n\nQuestion: {question}"
