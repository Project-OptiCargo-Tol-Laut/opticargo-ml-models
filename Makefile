.PHONY: install test lint generate-all train-all validate run docker-build clean-artifacts

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check src tests

generate-all:
	python -m opticargo_ml_models.training.generate_all --rows 8000

train-all:
	python -m opticargo_ml_models.training.train_all --release synthetic-baseline-v1

validate: test
	python -m compileall -q src tests
	ruff check src tests

run:
	uvicorn opticargo_ml_models.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t opticargo/opticargo-ml-models:dev .

clean-artifacts:
	python -c "from pathlib import Path; [p.unlink() for p in Path('artifacts').glob('*') if p.is_file() and p.name != '.gitkeep']"
