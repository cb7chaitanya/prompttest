.PHONY: test lint

test:
	.venv/bin/pytest -v

lint:
	.venv/bin/ruff check src/ tests/
