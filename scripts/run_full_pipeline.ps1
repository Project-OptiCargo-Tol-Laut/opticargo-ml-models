$ErrorActionPreference = "Stop"

Write-Host "Python:" (python -c "import sys; print(sys.executable)")
python --version

python -m pip install -e ".[dev]"
python -m pytest -q

python -m opticargo_ml_models.training.generate_all `
  --rows 8000 `
  --output-dir data/synthetic

python -m opticargo_ml_models.training.train_all `
  --data-dir data/synthetic `
  --artifact-dir artifacts `
  --release synthetic-baseline-v1

python -m pytest -q

$files = @(
  "artifacts/cargo_match_model.evaluation.json",
  "artifacts/demand_forecast_model.evaluation.json",
  "artifacts/anomaly_detector.evaluation.json"
)

$result = foreach ($file in $files) {
  $report = Get-Content $file -Raw | ConvertFrom-Json
  [PSCustomObject]@{
    Model = $report.model_name
    Version = $report.model_version
    Promotion = $report.promotion_decision
    Metrics = ($report.metrics | ConvertTo-Json -Compress)
  }
}

$result | Format-Table -AutoSize
