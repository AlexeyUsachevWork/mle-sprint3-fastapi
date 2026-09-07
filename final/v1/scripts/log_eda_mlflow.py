"""CLI: залогировать артефакты EDA в MLflow.

Перед запуском:
    python scripts/start_mlflow.py

Затем:
    python scripts/log_eda_mlflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.eda_logging import log_eda_to_mlflow  # noqa: E402


def main() -> int:
    try:
        log_eda_to_mlflow()
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
