#!/usr/bin/env bash
# Запуск API + Prometheus в Docker (модель и features — bind-mount с хоста).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
elif [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HOST_MODELS_DIR="${HOST_MODELS_DIR:-${ROOT_DIR}/models}"
export HOST_SERVING_DATA_DIR="${HOST_SERVING_DATA_DIR:-${ROOT_DIR}/data/serving}"
export HOST_CONFIGS_DIR="${HOST_CONFIGS_DIR:-${ROOT_DIR}/configs}"
export HOST_MONITORING_DIR="${HOST_MONITORING_DIR:-${ROOT_DIR}/monitoring}"

FEATURES="${HOST_SERVING_DATA_DIR}/clients_features.parquet"
MODEL="${HOST_MODELS_DIR}/model.bin"

if [[ ! -f "${MODEL}" ]]; then
  echo "Нет модели: ${MODEL}"
  echo "Сначала обучите: python -m src.models.train --config configs/train.yaml --skip-mlflow"
  exit 1
fi

if [[ ! -f "${FEATURES}" ]]; then
  echo "Нет features store — экспортирую из valid_features…"
  python -m scripts.export_serving_features \
    --source "${ROOT_DIR}/data/processed/datasets/valid_features.parquet" \
    --out "${FEATURES}"
fi

echo "HOST_MODELS_DIR=${HOST_MODELS_DIR}"
echo "HOST_SERVING_DATA_DIR=${HOST_SERVING_DATA_DIR}"
echo "Запуск docker compose (api + prometheus + grafana)…"
docker compose up -d --build api prometheus grafana

echo ""
echo "API:        http://127.0.0.1:${API_PORT:-8000}/docs"
echo "Health:     http://127.0.0.1:${API_PORT:-8000}/health"
echo "Metrics:    http://127.0.0.1:${API_PORT:-8000}/metrics"
echo "Prometheus: http://127.0.0.1:${PROMETHEUS_PORT:-9090}"
echo "Grafana:    http://127.0.0.1:${GRAFANA_PORT:-3000}  (user=${GRAFANA_USER:-admin})"
echo ""
echo "Пример:"
echo "  curl -s http://127.0.0.1:${API_PORT:-8000}/health"
echo "  curl -s -X POST http://127.0.0.1:${API_PORT:-8000}/recommend -H 'Content-Type: application/json' -d '{\"ncodpers\": <id>, \"top_k\": 7}'"
