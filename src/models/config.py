from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    name: str = "Qwen/Qwen3-0.6B"
    max_new_tokens: int = 512
    temperature: float = 0.0


class PathsConfig(StrictModel):
    raw_dir: str = "data/raw/vllm-0.10.1"
    processed_dir: str = "data/processed/"
    chunks_dir: str = "data/processed/chunks"
    index_dir: str = "data/processed/index"


class IndexingConfig(StrictModel):
    max_chunk_size: int = 2000
    chunk_overlap: int = 200
    retrieval_method: str = "bm25"
    k: int = 10


class GenerationConfig(StrictModel):
    max_context_length: int = 2000


class EvaluationConfig(StrictModel):
    recall_k: list[int] = [1, 3, 5, 10]
    overlap_threshold: float = 0.05


class Config(StrictModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
