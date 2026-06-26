from pathlib import Path
import yaml
from src.models import Config


def load_config(path: str | Path = "config.yml") -> Config:
    file_path = Path(path)
    if not file_path.is_file():
        return Config()

    try:
        with file_path.open("r") as stream:
            raw = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"Cannot read config {str(file_path)!r}: {e}") from e

    return Config.model_validate(raw)
