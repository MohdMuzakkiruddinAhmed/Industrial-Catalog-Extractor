.PHONY: install test lint format package demo

install:
	python -m pip install --require-hashes --requirement requirements-dev.lock
	python -m pip install --no-deps --editable .

test:
	python -m pytest

lint:
	python -m ruff check src tests scripts examples

format:
	python -m ruff format src tests scripts examples
	python -m ruff check --fix src tests scripts examples

package:
	python -m build

demo:
	python examples/create_sample_catalog.py --output .demo/corpus/sample-catalog.pdf
	industrial-catalog benchmark --config configs/pipeline.dry-run.yaml --output-dir .demo/run --dry-run --smoke
