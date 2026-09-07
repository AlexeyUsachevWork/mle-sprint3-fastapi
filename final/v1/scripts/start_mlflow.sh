#!/usr/bin/env bash
# Запуск MLflow Tracking Server + Model Registry.
# Backend/registry: PostgreSQL (личная БД курса), артефакты: S3 (Yandex Object Storage).
# Параметры доступов: .env.local или .env в корне репозитория (см. .env.example).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_FILE=""
if [[ -f "${ROOT_DIR}/.env.local" ]]; then
  ENV_FILE="${ROOT_DIR}/.env.local"
elif [[ -f "${ROOT_DIR}/.env" ]]; then
  ENV_FILE="${ROOT_DIR}/.env"
else
  echo "Не найден файл окружения: .env.local или .env" >&2
  echo "Скопируйте .env.example и заполните доступы курса." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${MLFLOW_S3_ENDPOINT_URL:?Задайте MLFLOW_S3_ENDPOINT_URL}"
: "${AWS_ACCESS_KEY_ID:?Задайте AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Задайте AWS_SECRET_ACCESS_KEY}"
: "${S3_BUCKET_NAME:?Задайте S3_BUCKET_NAME}"
: "${DB_DESTINATION_HOST:?Задайте DB_DESTINATION_HOST}"
: "${DB_DESTINATION_PORT:?Задайте DB_DESTINATION_PORT}"
: "${DB_DESTINATION_NAME:?Задайте DB_DESTINATION_NAME}"
: "${DB_DESTINATION_USER:?Задайте DB_DESTINATION_USER}"
: "${DB_DESTINATION_PASSWORD:?Задайте DB_DESTINATION_PASSWORD}"

export MLFLOW_S3_ENDPOINT_URL
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY

HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5000}"
BACKEND_URI="postgresql://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}?sslmode=require"

echo "Запуск MLflow Tracking Server"
echo "  env file:              ${ENV_FILE}"
echo "  UI:                    http://127.0.0.1:${PORT}"
echo "  backend / registry:    postgresql://***@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"
echo "  default-artifact-root: s3://${S3_BUCKET_NAME}"
echo "  S3 endpoint:           ${MLFLOW_S3_ENDPOINT_URL}"

exec mlflow server \
  --backend-store-uri "${BACKEND_URI}" \
  --registry-store-uri "${BACKEND_URI}" \
  --default-artifact-root "s3://${S3_BUCKET_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --no-serve-artifacts
