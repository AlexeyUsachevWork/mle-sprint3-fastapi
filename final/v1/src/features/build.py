"""Построение матрицы признаков из train/valid parquet.

Использование:
    python -m src.features.build --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import CATEGORICAL_COLS, DATE_COL, ID_COL, PRODUCT_COLS  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _slim_months(slim_dir: Path) -> list[str]:
    index_path = slim_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return sorted({p.name[:10] for p in slim_dir.glob("*_products.parquet")})


def attach_portfolio_lags(
    df: pd.DataFrame,
    slim_dir: Path,
    lags: list[int],
) -> pd.DataFrame:
    """Джойн портфеля на t−k из months_slim (полный месяц, не из сэмпла).

    Без утечки: только прошлое относительно fecha_dato строки.
    """
    if not lags:
        return df

    months = _slim_months(slim_dir)
    month_pos = {m: i for i, m in enumerate(months)}
    out = df.copy()
    out[DATE_COL] = out[DATE_COL].astype("string").str.slice(0, 10)
    product_cache: dict[str, pd.DataFrame] = {}

    def _products(month: str) -> pd.DataFrame:
        if month not in product_cache:
            path = slim_dir / f"{month}_products.parquet"
            prod = pd.read_parquet(path)
            cols = [ID_COL] + [p for p in PRODUCT_COLS if p in prod.columns]
            product_cache[month] = prod[cols]
        return product_cache[month]

    for lag in lags:
        pieces: list[pd.DataFrame] = []
        for d in out[DATE_COL].unique().tolist():
            d = str(d)
            ids = out.loc[out[DATE_COL] == d, [ID_COL]].drop_duplicates()
            idx = month_pos.get(d)
            block = ids.copy()
            if idx is None or idx < lag:
                for p in PRODUCT_COLS:
                    block[f"lag{lag}_has_{p}"] = np.int8(0)
                block[f"lag{lag}_missing"] = np.int8(1)
            else:
                src = months[idx - lag]
                prod = _products(src)
                merged = ids.merge(prod, on=ID_COL, how="left")
                miss = merged[PRODUCT_COLS[0]].isna() if PRODUCT_COLS[0] in merged.columns else pd.Series(True, index=merged.index)
                for p in PRODUCT_COLS:
                    if p in merged.columns:
                        block[f"lag{lag}_has_{p}"] = merged[p].fillna(0).astype("int8")
                    else:
                        block[f"lag{lag}_has_{p}"] = np.int8(0)
                block[f"lag{lag}_missing"] = miss.fillna(True).astype("int8")
            block[DATE_COL] = d
            pieces.append(block)

        lag_df = pd.concat(pieces, ignore_index=True)
        out = out.merge(lag_df, on=[ID_COL, DATE_COL], how="left")

        lag_cols = [f"lag{lag}_has_{p}" for p in PRODUCT_COLS]
        out[f"n_products_lag{lag}"] = out[lag_cols].sum(axis=1).astype("int16")

        # Недавние открытия t−1 → t (известны на момент прогноза)
        if lag == 1:
            opened_cols = []
            for p in PRODUCT_COLS:
                has_col = f"has_{p}"
                lag_col = f"lag{lag}_has_{p}"
                if has_col in out.columns:
                    oc = f"opened_lag1_{p}"
                    out[oc] = ((out[lag_col] == 0) & (out[has_col] == 1)).astype("int8")
                    opened_cols.append(oc)
            if opened_cols:
                out["n_opened_lag1"] = out[opened_cols].sum(axis=1).astype("int16")
            if "n_products" in out.columns:
                out["delta_n_products_lag1"] = (out["n_products"] - out[f"n_products_lag{lag}"]).astype("int16")

    # На случай дыр после merge
    fill_cols = [
        c
        for c in out.columns
        if c.startswith("lag")
        or c.startswith("opened_lag")
        or c.startswith("n_opened")
        or c.startswith("delta_n_products")
        or c.startswith("n_products_lag")
    ]
    for c in fill_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    return out


def fit_category_maps(train: pd.DataFrame, top_n_canal: int = 30) -> dict:
    """Частотное кодирование / топ-N для canal_entrada."""
    maps: dict = {"freq": {}, "canal_top": []}
    for col in CATEGORICAL_COLS:
        if col not in train.columns:
            continue
        freq = train[col].astype("string").value_counts(normalize=True)
        maps["freq"][col] = freq.to_dict()

    if "canal_entrada" in train.columns:
        top = (
            train["canal_entrada"]
            .astype("string")
            .value_counts()
            .head(top_n_canal)
            .index.tolist()
        )
        maps["canal_top"] = top
    return maps


def transform_frame(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    out = pd.DataFrame({ID_COL: df[ID_COL].to_numpy(), DATE_COL: df[DATE_COL].to_numpy()})

    # Числовые
    for col in ("age", "antiguedad", "renta", "ind_nuevo", "indrel", "ind_actividad_cliente", "cod_prov"):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    if "renta_missing" in df.columns:
        out["renta_missing"] = df["renta_missing"].fillna(0).astype("int8")
    else:
        out["renta_missing"] = 0

    # Частоты категорий
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        freq_map = maps.get("freq", {}).get(col, {})
        vals = df[col].astype("string")
        out[f"{col}_freq"] = vals.map(freq_map).fillna(0.0).astype("float32")

    if "canal_entrada" in df.columns:
        top = set(maps.get("canal_top") or [])
        canal = df["canal_entrada"].astype("string")
        out["canal_is_other"] = (~canal.isin(top)).astype("int8")

    # Портфель на t
    for p in PRODUCT_COLS:
        if p in df.columns:
            out[f"has_{p}"] = df[p].fillna(0).astype("int8")

    out["n_products"] = out[[c for c in out.columns if c.startswith("has_")]].sum(axis=1).astype("int16")

    # Таргеты
    for p in PRODUCT_COLS:
        tcol = f"target_{p}"
        if tcol in df.columns:
            out[tcol] = df[tcol].fillna(0).astype("int8")

    # Простые заполнения числовых медианой train позже; здесь fillna 0 для бустинга ок с индикатором
    num_cols = [c for c in out.columns if c not in {ID_COL, DATE_COL} and not c.startswith("target_")]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in {ID_COL, DATE_COL} and not c.startswith("target_")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение признаков")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "configs" / "train.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    processed = ROOT_DIR / (cfg.get("data") or {}).get("processed_dir", "data/processed")
    ds = processed / "datasets"
    slim_dir = processed / "eda" / "months_slim"
    feat_cfg = cfg.get("features") or {}
    lags = [int(x) for x in (feat_cfg.get("portfolio_lags") or [])]

    train_raw = pd.read_parquet(ds / "train.parquet")
    valid_raw = pd.read_parquet(ds / "valid.parquet")

    maps = fit_category_maps(train_raw)
    train_feat = transform_frame(train_raw, maps)
    valid_feat = transform_frame(valid_raw, maps)

    if lags:
        print(f"Лаги портфеля: {lags} (из {slim_dir})")
        train_feat = attach_portfolio_lags(train_feat, slim_dir, lags)
        valid_feat = attach_portfolio_lags(valid_feat, slim_dir, lags)

    feat_cols = feature_columns(train_feat)
    maps_path = ds / "category_maps.joblib"
    joblib.dump({"maps": maps, "feature_columns": feat_cols, "portfolio_lags": lags}, maps_path)

    train_path = ds / "train_features.parquet"
    valid_path = ds / "valid_features.parquet"
    train_feat.to_parquet(train_path, index=False)
    valid_feat.to_parquet(valid_path, index=False)

    meta = {
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "portfolio_lags": lags,
        "train_rows": int(len(train_feat)),
        "valid_rows": int(len(valid_feat)),
    }
    (ds / "features_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Готово:")
    print(f"  features: {len(feat_cols)}")
    print(f"  portfolio_lags: {lags}")
    print(f"  train: {train_path} ({len(train_feat):,})")
    print(f"  valid: {valid_path} ({len(valid_feat):,})")
    print(f"  maps:  {maps_path}")


if __name__ == "__main__":
    main()
