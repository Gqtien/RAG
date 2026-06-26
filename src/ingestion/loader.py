from pathlib import Path

ALLOWED_SUFFIXES: list[str] = [".py", ".md", ".txt"]


def load_files(path: str | Path) -> dict[str, str]:
    root = Path(path)
    files_dump: dict[str, str] = {}

    for file in list_files(root):
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        files_dump[str(file.relative_to(root))] = text
    return files_dump


def list_files(root: Path) -> list[Path]:
    return [
        file
        for file in root.rglob("*")
        if file.is_file() and file.suffix in ALLOWED_SUFFIXES
    ]
