# Запуск API + Prometheus в Docker (Windows PowerShell).
# Модель и features — bind-mount с хоста.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Test-Path ".env.local") {
  Get-Content ".env.local" | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $k, $v = $_.Split("=", 2)
    if ($k -and $v) { Set-Item -Path "Env:$($k.Trim())" -Value $v.Trim() }
  }
}

$env:HOST_MODELS_DIR = if ($env:HOST_MODELS_DIR) { $env:HOST_MODELS_DIR } else { Join-Path $Root "models" }
$env:HOST_SERVING_DATA_DIR = if ($env:HOST_SERVING_DATA_DIR) { $env:HOST_SERVING_DATA_DIR } else { Join-Path $Root "data\serving" }
$env:HOST_CONFIGS_DIR = if ($env:HOST_CONFIGS_DIR) { $env:HOST_CONFIGS_DIR } else { Join-Path $Root "configs" }
$env:HOST_MONITORING_DIR = if ($env:HOST_MONITORING_DIR) { $env:HOST_MONITORING_DIR } else { Join-Path $Root "monitoring" }

$Model = Join-Path $env:HOST_MODELS_DIR "model.bin"
$Features = Join-Path $env:HOST_SERVING_DATA_DIR "clients_features.parquet"

if (-not (Test-Path $Model)) {
  Write-Error "Нет модели: $Model"
}

if (-not (Test-Path $Features)) {
  Write-Host "Экспорт features store…"
  $py = if (Test-Path "C:\work\yandex\source\mle-uplift-final-project\uplift_env\Scripts\python.exe") {
    "C:\work\yandex\source\mle-uplift-final-project\uplift_env\Scripts\python.exe"
  } else { "python" }
  & $py -m scripts.export_serving_features --out $Features
}

Write-Host "HOST_MODELS_DIR=$($env:HOST_MODELS_DIR)"
Write-Host "HOST_SERVING_DATA_DIR=$($env:HOST_SERVING_DATA_DIR)"
docker compose up -d --build api prometheus grafana

$port = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$gport = if ($env:GRAFANA_PORT) { $env:GRAFANA_PORT } else { "3000" }
Write-Host ""
Write-Host "API:        http://127.0.0.1:$port/docs"
Write-Host "Health:     http://127.0.0.1:$port/health"
Write-Host "Prometheus: http://127.0.0.1:$(if ($env:PROMETHEUS_PORT) {$env:PROMETHEUS_PORT} else {'9090'})"
Write-Host "Grafana:    http://127.0.0.1:$gport  (admin / `$env:GRAFANA_PASSWORD or admin)"
