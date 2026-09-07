"""Запуск MLflow Tracking Server (удобно на Windows).

Читает доступы из `.env.local` или `.env` в корне репозитория.
Схема как в спринте 2: PostgreSQL (backend + registry) и S3 (артефакты).

Использование:
    python scripts/start_mlflow.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_env() -> Path:
    for name in (".env.local", ".env"):
        path = ROOT_DIR / name
        if path.exists():
            load_dotenv(path, override=True)
            return path
    raise FileNotFoundError(
        "Не найден .env.local или .env. Скопируйте .env.example и заполните доступы."
    )


def main() -> int:
    env_path = _load_env()

    required = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "S3_BUCKET_NAME",
        "DB_DESTINATION_HOST",
        "DB_DESTINATION_PORT",
        "DB_DESTINATION_NAME",
        "DB_DESTINATION_USER",
        "DB_DESTINATION_PASSWORD",
    ]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        print(f"Не заданы переменные: {', '.join(missing)}", file=sys.stderr)
        return 1

    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
    host = os.environ.get("MLFLOW_HOST", "0.0.0.0")
    port = os.environ.get("MLFLOW_PORT", "5000")

    backend_uri = (
        f"postgresql://{os.environ['DB_DESTINATION_USER']}:"
        f"{os.environ['DB_DESTINATION_PASSWORD']}@"
        f"{os.environ['DB_DESTINATION_HOST']}:{os.environ['DB_DESTINATION_PORT']}/"
        f"{os.environ['DB_DESTINATION_NAME']}?sslmode=require"
    )
    bucket = os.environ["S3_BUCKET_NAME"]

    print("Запуск MLflow Tracking Server")
    print(f"  env file:              {env_path}")
    print(f"  UI:                    http://127.0.0.1:{port}")
    print(
        "  backend / registry:    "
        f"postgresql://***@{os.environ['DB_DESTINATION_HOST']}:"
        f"{os.environ['DB_DESTINATION_PORT']}/{os.environ['DB_DESTINATION_NAME']}"
    )
    print(f"  default-artifact-root: s3://{bucket}")
    print(f"  S3 endpoint:           {os.environ['MLFLOW_S3_ENDPOINT_URL']}")
    sys.stdout.flush()

    # На Windows mlflow может вызывать waitress-serve из PATH рядом с python.exe
    venv_scripts = Path(sys.executable).resolve().parent
    env = os.environ.copy()
    env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")

    return subprocess.call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            backend_uri,
            "--registry-store-uri",
            backend_uri,
            "--default-artifact-root",
            f"s3://{bucket}",
            "--host",
            host,
            "--port",
            port,
            "--no-serve-artifacts",
        ],
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
