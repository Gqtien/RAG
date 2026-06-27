VENV := .venv
PY   := $(VENV)/bin/python
ARGS = awk 'BEGIN{RS=ORS="\0"} f&&$$0!="--"; $$0=="$@"{f=1}' /proc/$$PPID/cmdline | xargs -0

usage:
	@echo 'Usage: make run <command> [args...]'
	@echo 'e.g.   make run search "comment on fait stp" --k 10'

run: $(VENV)
	@$(ARGS) uv run python -m src || true

debug: $(VENV)
	@$(ARGS) uv run python -m pdb -m src || true

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

%: FORCE
	@:

FORCE:

.PHONY: usage run debug install lint lint-strict clean FORCE
