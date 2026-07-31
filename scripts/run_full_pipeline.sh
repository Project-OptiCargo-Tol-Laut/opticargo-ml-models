#!/usr/bin/env bash
set -euo pipefail

python -c 'import sys; print(sys.executable)'
python --version
python -m pip install -e '.[dev]'
python -m pytest -q
python -m opticargo_ml_models.training.generate_all --rows 8000 --output-dir data/synthetic
python -m opticargo_ml_models.training.train_all \
  --data-dir data/synthetic \
  --artifact-dir artifacts \
  --release synthetic-baseline-v1
python -m pytest -q
