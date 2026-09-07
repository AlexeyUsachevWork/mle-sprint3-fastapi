"""Подготовка датасета: чистка профиля, пары t→t+1, time-split.

Использование:
    python -m src.data.prepare --config configs/train.yaml

Опирается на помесячные slim-продукты из EDA (`data/processed/eda/months_slim/`).
Профиль клиента дособирается одним проходом по CSV только для нужных месяцев.
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

from src.constants import CATEGORICAL_COLS, DATE_COL, ID_COL, PRODUCT_COLS, PROFILE_COLS  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_numeric_clean(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "NA": pd.NA, "na": pd.NA, "None": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def clean_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("age", "antiguedad", "renta"):
        if col in out.columns:
            out[col] = to_numeric_clean(out[col])

    if "age" in out.columns:
        out.loc[(out["age"] < 0) | (out["age"] > 100), "age"] = np.nan
    if "antiguedad" in out.columns:
        out.loc[(out["antiguedad"] < 0) | (out["antiguedad"] > 300), "antiguedad"] = np.nan

    if "renta" in out.columns:
        out["renta_missing"] = out["renta"].isna().astype("int8")
        out["renta"] = np.log1p(out["renta"].clip(lower=0))

    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip().fillna("UNK")

    for col in ("ind_nuevo", "indrel", "ind_actividad_cliente", "cod_prov"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _slim_index(slim_dir: Path) -> list[str]:
    index_path = slim_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return sorted({p.name[:10] for p in slim_dir.glob("*_products.parquet")})


def build_target_panel(slim_dir: Path, months: list[str]) -> pd.DataFrame:
    """Для каждого t строит портфель на t и таргеты 0→1 из t+1."""
    rows: list[pd.DataFrame] = []
    for t, t1 in zip(months, months[1:]):
        path_t = slim_dir / f"{t}_products.parquet"
        path_t1 = slim_dir / f"{t1}_products.parquet"
        if not path_t.exists() or not path_t1.exists():
            raise FileNotFoundError(f"Нет slim-продуктов для пары {t} → {t1}. Сначала прогоните EDA.")

        df_t = pd.read_parquet(path_t)
        df_t1 = pd.read_parquet(path_t1)
        merged = df_t.merge(df_t1, on=ID_COL, how="inner", suffixes=("", "_t1"))

        out = merged[[ID_COL]].copy()
        out[DATE_COL] = t
        for p in PRODUCT_COLS:
            if p not in merged.columns:
                continue
            left = merged[p].fillna(0).astype("int8")
            right = merged[f"{p}_t1"].fillna(0).astype("int8")
            out[p] = left  # портфель на t
            out[f"target_{p}"] = ((left == 0) & (right == 1)).astype("int8")

        rows.append(out)
        del df_t, df_t1, merged, out
        gc.collect()
        print(f"  targets {t} → {t1}: ok")

    return pd.concat(rows, ignore_index=True)


def extract_profiles(
    raw_path: Path,
    months_needed: set[str],
    chunksize: int,
    out_dir: Path,
) -> None:
    """Один проход CSV → parquet профиля на месяц (только нужные даты)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    usecols = [ID_COL, DATE_COL] + [c for c in PROFILE_COLS]
    header = pd.read_csv(raw_path, nrows=0).columns.tolist()
    usecols = [c for c in usecols if c in header]

    buffers: dict[str, list[pd.DataFrame]] = {m: [] for m in months_needed}
    reader = pd.read_csv(raw_path, usecols=usecols, chunksize=chunksize, low_memory=False)

    for i, chunk in enumerate(reader):
        chunk[DATE_COL] = chunk[DATE_COL].astype("string").str.slice(0, 10)
        mask = chunk[DATE_COL].isin(months_needed)
        if not mask.any():
            continue
        part = chunk.loc[mask].copy()
        part = clean_profile(part)
        for d, g in part.groupby(DATE_COL, sort=False):
            if d in buffers:
                buffers[d].append(g.drop(columns=[DATE_COL]))
        if (i + 1) % 5 == 0:
            print(f"  profile chunks={i + 1}")
        del chunk, part
        gc.collect()

    for d, parts in buffers.items():
        if not parts:
            print(f"  WARN: нет профиля для {d}")
            continue
        df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=[ID_COL], keep="last")
        path = out_dir / f"{d}_profile.parquet"
        df.to_parquet(path, index=False)
        print(f"  profile {d}: {len(df):,} rows → {path.name}")
        del df
        gc.collect()


def sample_panel(df: pd.DataFrame, n_per_month: int | None, seed: int) -> pd.DataFrame:
    if not n_per_month or n_per_month <= 0:
        return df
    parts = []
    rng = np.random.default_rng(seed)
    for d, g in df.groupby(DATE_COL, sort=False):
        if len(g) <= n_per_month:
            parts.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=n_per_month, replace=False)
            parts.append(g.loc[idx])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка данных рекомендателя")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "configs" / "train.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data") or {}
    split_cfg = cfg.get("split") or {}
    seed = int(cfg.get("random_seed", 42))

    raw_path = ROOT_DIR / data_cfg.get("raw_path", "data/raw/train_ver2.csv")
    processed = ROOT_DIR / data_cfg.get("processed_dir", "data/processed")
    slim_dir = processed / "eda" / "months_slim"
    profile_dir = processed / "profiles"
    out_dir = processed / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunksize = int(data_cfg.get("chunksize", 400_000))
    n_train = data_cfg.get("clients_per_month_train", 60_000)
    n_valid = data_cfg.get("clients_per_month_valid", 80_000)

    if not raw_path.exists():
        raise FileNotFoundError(f"Нет сырого файла: {raw_path}")

    months = _slim_index(slim_dir)
    if len(months) < 2:
        raise RuntimeError("Недостаточно месяцев в months_slim — сначала выполните EDA.")

    train_end = split_cfg.get("train_end", "2015-12-28")
    valid_end = split_cfg.get("valid_end", "2016-04-28")
    pairable = months[:-1]

    train_months = [m for m in pairable if m <= train_end]
    valid_months = [m for m in pairable if train_end < m <= valid_end]
    if not train_months or not valid_months:
        raise RuntimeError(f"Пустой split: train={train_months}, valid={valid_months}")

    print("1) Целевые панели из slim-продуктов…")
    panel = build_target_panel(slim_dir, months)
    panel = panel[panel[DATE_COL].isin(train_months + valid_months)].copy()
    print(f"   panel rows: {len(panel):,}")

    print("2) Профили клиентов для месяцев train/valid…")
    need_profiles = {m for m in train_months + valid_months if not (profile_dir / f"{m}_profile.parquet").exists()}
    if need_profiles:
        extract_profiles(raw_path, set(train_months + valid_months), chunksize, profile_dir)
    else:
        print("   профили уже есть — пропускаем чтение CSV")

    print("3) Джойн профиля + сэмплирование split…")
    parts = []
    for d, g in panel.groupby(DATE_COL, sort=False):
        prof_path = profile_dir / f"{d}_profile.parquet"
        if not prof_path.exists():
            raise FileNotFoundError(prof_path)
        prof = pd.read_parquet(prof_path)
        merged = g.merge(prof, on=ID_COL, how="inner")
        parts.append(merged)
        del prof, merged
        gc.collect()
    full = pd.concat(parts, ignore_index=True)
    del parts, panel
    gc.collect()

    train_df = sample_panel(full[full[DATE_COL].isin(train_months)], n_train, seed)
    valid_df = sample_panel(full[full[DATE_COL].isin(valid_months)], n_valid, seed + 1)

    train_path = out_dir / "train.parquet"
    valid_path = out_dir / "valid.parquet"
    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)

    meta = {
        "train_months": train_months,
        "valid_months": valid_months,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "n_products": len(PRODUCT_COLS),
        "clients_per_month_train": n_train,
        "clients_per_month_valid": n_valid,
        "random_seed": seed,
    }
    meta_path = out_dir / "prepare_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Готово:")
    print(f"  train: {train_path} ({len(train_df):,})")
    print(f"  valid: {valid_path} ({len(valid_df):,})")
    print(f"  meta:  {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
