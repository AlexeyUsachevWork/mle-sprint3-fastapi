"""Подготовка датасета со сплитом по пользователям (не по времени).

Параллельный эксперимент к `src.data.prepare` (time-split).
Месяцы одни и те же для train/valid; клиенты разбиваются на непересекающиеся множества.

Использование:
    python -m src.data.prepare_user_split --config configs/train_user_split.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import DATE_COL, ID_COL, PRODUCT_COLS  # noqa: E402
from src.data.prepare import (  # noqa: E402
    _slim_index,
    build_target_panel,
    extract_profiles,
    sample_panel,
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка данных: user-split")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "configs" / "train_user_split.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data") or {}
    split_cfg = cfg.get("split") or {}
    seed = int(cfg.get("random_seed", 42))

    raw_path = ROOT_DIR / data_cfg.get("raw_path", "data/raw/train_ver2.csv")
    processed = ROOT_DIR / data_cfg.get("processed_dir", "data/processed")
    slim_dir = processed / "eda" / "months_slim"
    profile_dir = processed / "profiles"
    out_dir = processed / data_cfg.get("datasets_dir", "datasets_user_split")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunksize = int(data_cfg.get("chunksize", 400_000))
    n_train = data_cfg.get("clients_per_month_train", 100_000)
    n_valid = data_cfg.get("clients_per_month_valid", 133_333)
    valid_frac = float(split_cfg.get("valid_user_frac", 0.2))
    months_start = split_cfg.get("months_start", "2015-01-28")
    months_end = split_cfg.get("months_end", "2016-04-28")

    if not raw_path.exists():
        raise FileNotFoundError(f"Нет сырого файла: {raw_path}")

    months = _slim_index(slim_dir)
    if len(months) < 2:
        raise RuntimeError("Недостаточно месяцев в months_slim — сначала выполните EDA.")

    pairable = months[:-1]
    used_months = [m for m in pairable if months_start <= m <= months_end]
    if len(used_months) < 2:
        raise RuntimeError(f"Слишком мало месяцев для user-split: {used_months}")

    print(f"1) Целевые панели (user-split, months={used_months[0]}…{used_months[-1]})…")
    panel = build_target_panel(slim_dir, months)
    panel = panel[panel[DATE_COL].isin(used_months)].copy()
    print(f"   panel rows: {len(panel):,}")

    print("2) Профили клиентов…")
    need_profiles = {m for m in used_months if not (profile_dir / f"{m}_profile.parquet").exists()}
    if need_profiles:
        extract_profiles(raw_path, set(used_months), chunksize, profile_dir)
    else:
        print("   профили уже есть — пропускаем чтение CSV")

    print("3) Джойн профиля…")
    parts = []
    for d, g in panel.groupby(DATE_COL, sort=False):
        prof_path = profile_dir / f"{d}_profile.parquet"
        if not prof_path.exists():
            raise FileNotFoundError(prof_path)
        prof = pd.read_parquet(prof_path)
        parts.append(g.merge(prof, on=ID_COL, how="inner"))
        del prof
        gc.collect()
    full = pd.concat(parts, ignore_index=True)
    del parts, panel
    gc.collect()

    print("4) Сплит клиентов (ncodpers)…")
    clients = full[ID_COL].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(clients)
    n_valid_clients = max(1, int(round(len(clients) * valid_frac)))
    valid_ids = set(clients[:n_valid_clients].tolist())
    train_ids = set(clients[n_valid_clients:].tolist())
    assert not (train_ids & valid_ids)

    train_pool = full[full[ID_COL].isin(train_ids)]
    valid_pool = full[full[ID_COL].isin(valid_ids)]
    del full
    gc.collect()

    train_df = sample_panel(train_pool, n_train, seed)
    valid_df = sample_panel(valid_pool, n_valid, seed + 1)
    del train_pool, valid_pool
    gc.collect()

    # Гарантия: после сэмпла пересечения клиентов нет
    overlap = set(train_df[ID_COL].unique()) & set(valid_df[ID_COL].unique())
    if overlap:
        raise RuntimeError(f"Пересечение клиентов после сэмпла: {len(overlap)}")

    train_path = out_dir / "train.parquet"
    valid_path = out_dir / "valid.parquet"
    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)

    meta = {
        "split_mode": "user",
        "months": used_months,
        "valid_user_frac": valid_frac,
        "n_train_clients_pool": len(train_ids),
        "n_valid_clients_pool": len(valid_ids),
        "n_train_clients_sample": int(train_df[ID_COL].nunique()),
        "n_valid_clients_sample": int(valid_df[ID_COL].nunique()),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "n_products": len(PRODUCT_COLS),
        "clients_per_month_train": n_train,
        "clients_per_month_valid": n_valid,
        "random_seed": seed,
        "note": "Контрольный эксперимент: те же месяцы в train/valid, сплит по ncodpers. "
        "Возможен leakage временных паттернов относительно time-split.",
    }
    meta_path = out_dir / "prepare_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Готово (user-split):")
    print(f"  train: {train_path} ({len(train_df):,})")
    print(f"  valid: {valid_path} ({len(valid_df):,})")
    print(f"  meta:  {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
