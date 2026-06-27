import sys
import fire
from src.cli import RagCLI

if __name__ == "__main__":
    try:
        fire.Fire(RagCLI)
    except Exception as exc:
        sys.exit(f"Error: {exc}")
