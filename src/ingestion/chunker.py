from src.models import Chunk, IndexingConfig


def chunk_files(config: IndexingConfig, files: dict[str, str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    size, overlap = config.max_chunk_size, config.chunk_overlap

    for path, content in files.items():
        for start in range(0, len(content), size - overlap):
            stop = min(start + size, len(content))
            chunks.append(
                Chunk(
                    file_path=path,
                    first_character_index=start,
                    last_character_index=stop,
                    text=content[start:stop],
                )
            )
            if stop == len(content):
                break
    return chunks
