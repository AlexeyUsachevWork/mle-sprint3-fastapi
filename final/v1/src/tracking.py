"""Подключение к MLflow Tracking Server (как в спринте 2)."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_project_env() -> Path | None:
    """Загружает `.env.local` или `.env` из корня репозитория."""
    for name in (".env.local", ".env"):
        path = ROOT_DIR / name
        if path.exists():
            load_dotenv(path, override=True)
            return path
    return None


def setup_mlflow(tracking_uri: str | None = None) -> str:
    """Настраивает tracking/registry URI и S3 endpoint для артефактов."""
    load_project_env()
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")

    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(key):
            raise RuntimeError(
                f"Не задана переменная окружения: {key}. "
                "Проверьте .env.local / .env (см. .env.example)."
            )

    if tracking_uri is None:
        config_path = ROOT_DIR / "configs" / "train.yaml"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            tracking_uri = (cfg.get("mlflow") or {}).get("tracking_uri")
        tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return tracking_uri


def get_or_create_experiment(name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return mlflow.create_experiment(name)
    return experiment.experiment_id


def load_train_config() -> dict:
    config_path = ROOT_DIR / "configs" / "train.yaml"
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
