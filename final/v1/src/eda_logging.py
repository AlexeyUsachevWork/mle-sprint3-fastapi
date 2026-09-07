"""Логирование результатов EDA в MLflow (библиотечный модуль).

Ориентир: mle-project-sprint-2-v001/model_improvement/stage_02_eda.py

Перед запуском должен быть поднят Tracking Server:
    python scripts/start_mlflow.py

Использование:
    python scripts/log_eda_mlflow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd

from src.tracking import ROOT_DIR, get_or_create_experiment, load_train_config, setup_mlflow

EDA_DIR = ROOT_DIR / "data" / "processed" / "eda"
NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "01_eda.ipynb"

# Компактные артефакты (без тяжёлых months_slim / profile_sample)
SUMMARY_FILES = [
    "01_schema_sample.parquet",
    "02_dtype_map.json",
    "03_month_counts.parquet",
    "03_na_rates.parquet",
    "03_pass_summary.json",
    "03_product_ownership.parquet",
    "05_clients_per_month.parquet",
    "05_month_overlap.parquet",
    "06_categorical_freq.parquet",
    "07_product_ownership_sample.parquet",
    "07_product_corr.parquet",
    "08_transitions_by_month.parquet",
    "08_new_by_product_month.parquet",
    "09_cleaning_plan.json",
    "10_split_suggestion.json",
    "11_eda_summary.json",
]

PLOT_FILES = [
    "04_rows_by_month.png",
    "05_clients_by_month.png",
    "06_numeric_profile_hist.png",
    "07_ownership_bar.png",
    "07_product_corr_heatmap.png",
    "08_share_clients_with_new.png",
    "08_new_products_total.png",
]


def _require_summary() -> dict:
    path = EDA_DIR / "11_eda_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Нет {path}. Сначала выполните notebooks/01_eda.ipynb "
            "или убедитесь, что локальные артефакты EDA на месте."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _build_conclusions(summary: dict) -> list[str]:
    top = summary.get("top5_owned_products") or []
    top_txt = ", ".join(
        f"{x['product']} ({float(x['ownership_rate']):.1%})" for x in top[:3]
    )
    return [
        (
            f"Данные — ежемесячные срезы клиентов: {summary.get('n_months')} месяцев "
            f"({summary.get('date_min')} … {summary.get('date_max')}), "
            f"всего ~{int(summary.get('n_rows_total', 0)):,} строк."
        ),
        (
            "Целевое действие — появление нового продукта (переход 0→1 между соседними месяцами), "
            "а не подтверждение уже имеющегося портфеля."
        ),
        (
            f"Кросс-сейл редкий: в среднем ~{float(summary.get('avg_share_clients_with_new', 0)):.1%} "
            f"клиентов открывают хотя бы один новый продукт; "
            f"среднее число новых продуктов на клиента ≈ {float(summary.get('avg_mean_new_per_client', 0)):.3f}."
        ),
        (
            "Метрика accuracy по всем флагам продуктов бесполезна из‑за массы нулей. "
            "Оцениваем короткий список рекомендаций: MAP@7, Precision@7, Recall@7."
        ),
        (
            f"Владение продуктами сильно дисбалансно (топ: {top_txt}). "
            "Открытия лидируют иначе (recibo, зарплатный/пенсионный, карта) — "
            "модель может перекоситься к массовым продуктам."
        ),
        (
            "Клиенты соседних месяцев пересекаются почти полностью (~99.7%) — "
            "валидация только по времени (fecha_dato), без случайного split по строкам."
        ),
        (
            "Почти пустые поля conyuemp / ult_fec_cli_1t не используем; "
            "renta имеет ~20% пропусков; age/antiguedad/renta требуют чистки из текста."
        ),
    ]


def _write_markdown(conclusions: list[str], summary: dict, path: Path) -> None:
    lines = [
        "# Результаты EDA — рекомендации банковских продуктов",
        "",
        "## Краткие цифры",
        "",
        f"- Месяцев: **{summary.get('n_months')}** ({summary.get('date_min')} … {summary.get('date_max')})",
        f"- Строк: **{int(summary.get('n_rows_total', 0)):,}**",
        f"- Доля клиентов с ≥1 новым продуктом: **{float(summary.get('avg_share_clients_with_new', 0)):.2%}**",
        f"- Среднее число новых продуктов на клиента: **{float(summary.get('avg_mean_new_per_client', 0)):.4f}**",
        "",
        "## Выводы",
        "",
    ]
    for i, text in enumerate(conclusions, start=1):
        lines.append(f"{i}. {text}")
        lines.append("")
    lines.extend(
        [
            "## Артефакты",
            "",
            "- Ноутбук: `notebooks/01_eda.ipynb`",
            "- Локальные файлы: `data/processed/eda/`",
            "- В MLflow: каталог артефактов `eda/` текущего run",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _log_metrics_from_tables() -> None:
    """Логирует ключевые числовые метрики EDA в run."""
    summary = _require_summary()
    mlflow.log_metric("n_rows_total", float(summary["n_rows_total"]))
    mlflow.log_metric("n_months", float(summary["n_months"]))
    mlflow.log_metric("avg_share_clients_with_new", float(summary["avg_share_clients_with_new"]))
    mlflow.log_metric("avg_mean_new_per_client", float(summary["avg_mean_new_per_client"]))

    overlap_path = EDA_DIR / "05_month_overlap.parquet"
    if overlap_path.exists():
        overlap = pd.read_parquet(overlap_path)
        mlflow.log_metric("avg_month_overlap_rate", float(overlap["overlap_rate_of_t"].mean()))

    tr_path = EDA_DIR / "08_transitions_by_month.parquet"
    if tr_path.exists():
        tr = pd.read_parquet(tr_path)
        mlflow.log_metric("median_p99_new_products", float(tr["p99_new"].median()))


def log_eda_to_mlflow() -> str:
    # Windows-консоль часто в cp1251 — MLflow при завершении run печатает unicode-символы
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_train_config()
    mlflow_cfg = cfg.get("mlflow") or {}
    experiment_name = mlflow_cfg.get("experiment_name", "bank-product-recommender")
    run_name = mlflow_cfg.get("eda_run_name", "bank-rec-eda-001")

    tracking_uri = setup_mlflow(mlflow_cfg.get("tracking_uri"))
    summary = _require_summary()
    conclusions = _build_conclusions(summary)

    report_path = EDA_DIR / "eda_results.md"
    conclusions_path = EDA_DIR / "conclusions.txt"
    _write_markdown(conclusions, summary, report_path)
    conclusions_path.write_text("\n".join(conclusions), encoding="utf-8")

    experiment_id = get_or_create_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        mlflow.set_tags(
            {
                "stage": "eda",
                "has_model": "false",
                "project": "bank-product-recommender",
                "notebook": "notebooks/01_eda.ipynb",
            }
        )
        mlflow.log_params(
            {
                "n_months": int(summary["n_months"]),
                "date_min": summary["date_min"],
                "date_max": summary["date_max"],
                "train_end": (cfg.get("split") or {}).get("train_end"),
                "valid_end": (cfg.get("split") or {}).get("valid_end"),
                "top_k": (cfg.get("model") or {}).get("top_k"),
                "random_seed": cfg.get("random_seed"),
            }
        )
        _log_metrics_from_tables()
        mlflow.log_dict(summary, "eda/eda_summary.json")

        mlflow.log_artifact(str(report_path), artifact_path="eda")
        mlflow.log_artifact(str(conclusions_path), artifact_path="eda")
        if NOTEBOOK_PATH.exists():
            mlflow.log_artifact(str(NOTEBOOK_PATH), artifact_path="eda")

        for name in SUMMARY_FILES:
            path = EDA_DIR / name
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="eda/tables")

        for name in PLOT_FILES:
            path = EDA_DIR / name
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="eda/plots")

        print("EDA залогирован в MLflow")
        print(f"  tracking   : {tracking_uri}")
        print(f"  experiment : {experiment_name}")
        print(f"  run        : {run_name} ({run.info.run_id})")
        print(f"  UI         : {tracking_uri.rstrip('/')}/#/experiments/{experiment_id}")
        return run.info.run_id


def main() -> int:
    try:
        log_eda_to_mlflow()
    except Exception as exc:  # noqa: BLE001 — понятное сообщение в CLI
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
