VENV := .venv
PY   := $(VENV)/bin/python

CMDS := run debug install lint lint-strict clean
ARGS := $(filter-out $(CMDS),$(MAKECMDGOALS))

usage:
	@echo "Usage: make run [config_file]"

run: $(VENV)
	@uv run python -m src $(ARGS)

debug: $(VENV)
	@uv run python -m pdb -m src $(ARGS)

install $(VENV):
	@uv sync

lint: $(VENV)
	-@$(PY) -m flake8 src
	-@$(PY) -m mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: $(VENV)
	-@$(PY) -m flake8 src
	-@$(PY) -m mypy src --strict

clean:
	@rm -rf $(VENV)
	@find . -type d \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) -exec rm -rf {} +

$(ARGS):
	@:

.PHONY: $(CMDS)
