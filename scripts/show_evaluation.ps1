$files = @(
  "artifacts/cargo_match_model.evaluation.json",
  "artifacts/demand_forecast_model.evaluation.json",
  "artifacts/anomaly_detector.evaluation.json"
)

foreach ($file in $files) {
  if (-not (Test-Path $file)) {
    Write-Warning "Belum ada: $file"
    continue
  }
  $report = Get-Content $file -Raw | ConvertFrom-Json
  Write-Host "`n=== $($report.model_name) / $($report.model_version) ==="
  $report.metrics | Format-List
  Write-Host "Promotion: $($report.promotion_decision) - $($report.promotion_reason)"
}
