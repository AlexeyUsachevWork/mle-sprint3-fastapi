"""Экспорт таблицы признаков для online serving (lookup по ncodpers).

Использование:
    python -m scripts.export_serving_features
    python -m scripts.export_serving_features --source data/processed/datasets/valid_features.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import DATE_COL, ID_COL  # noqa: E402


def export_features(source: Path, out_path: Path) -> dict:
    df = pd.read_parquet(source)
    if ID_COL not in df.columns:
        raise KeyError(f"Нет {ID_COL} в {source}")

    before = len(df)
    if DATE_COL in df.columns:
        df = df.sort_values(DATE_COL).groupby(ID_COL, as_index=False).tail(1)
        month = str(df[DATE_COL].astype(str).str.slice(0, 10).mode().iloc[0]) if len(df) else None
    else:
        df = df.drop_duplicates(subset=[ID_COL], keep="last")
        month = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    meta = {
        "source": str(source),
        "output": str(out_path),
        "rows_in": before,
        "clients": int(len(df)),
        "snapshot_month": month,
    }
    (out_path.parent / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт features store для API")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT_DIR / "data" / "processed" / "datasets" / "valid_features.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT_DIR / "data" / "serving" / "clients_features.parquet",
    )
    args = parser.parse_args()
    meta = export_features(args.source, args.out)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
