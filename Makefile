.PHONY: setup check lint format typecheck test build smoke

setup:
	uv sync --locked
	uv run --no-sync pre-commit install

check: lint typecheck test

lint:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .

format:
	uv run --no-sync ruff check --fix .
	uv run --no-sync ruff format .

typecheck:
	uv run --no-sync pyright

test:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync pytest

build:
	uv build --no-build-isolation

smoke: build
	bash tools/wheel_smoke.sh
