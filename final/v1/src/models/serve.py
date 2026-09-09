"""Онлайн-инференс: загрузка model.bin + рекомендации top-K.

Не импортирует train.py (там MLflow) — удобно для slim Docker-образа сервиса.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.constants import DATE_COL, ID_COL, PRODUCT_COLS


def _feature_matrix(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    return df[feat_cols].to_numpy(dtype=np.float32)


def _owned_matrix(df: pd.DataFrame, products: list[str]) -> np.ndarray:
    mats = []
    for p in products:
        col = f"has_{p}"
        if col in df.columns:
            mats.append((df[col].to_numpy() == 1))
        else:
            mats.append(np.zeros(len(df), dtype=bool))
    return np.vstack(mats).T


def _bayes_shrink_scores(
    p_hat: np.ndarray,
    pop: float,
    n_pos: int,
    prior_strength: float,
) -> np.ndarray:
    m = float(prior_strength)
    n = float(max(int(n_pos), 0))
    if m <= 0:
        return p_hat.astype(np.float32, copy=False)
    if n <= 0:
        return np.full_like(p_hat, float(pop), dtype=np.float32)
    return ((n * p_hat + m * float(pop)) / (n + m)).astype(np.float32)


def _predict_scores(
    models: dict,
    df: pd.DataFrame,
    feat_cols: list[str],
    popularity: dict[str, float],
    products: list[str],
    train_positives: dict[str, int] | None = None,
    bayes_prior: float | None = None,
) -> np.ndarray:
    x = _feature_matrix(df, feat_cols)
    n = len(df)
    scores = np.zeros((n, len(products)), dtype=np.float32)
    pos_map = train_positives or {}
    use_shrink = bayes_prior is not None and float(bayes_prior) > 0
    for j, p in enumerate(products):
        pop = float(popularity.get(p, 0.0))
        model = models.get(p)
        if model is None:
            scores[:, j] = pop
            continue
        p_hat = model.predict_proba(x)[:, 1]
        if use_shrink:
            scores[:, j] = _bayes_shrink_scores(p_hat, pop, int(pos_map.get(p, 0)), float(bayes_prior))
        else:
            scores[:, j] = p_hat
    return scores


def _rank_fast(
    owned: np.ndarray,
    scores: np.ndarray,
    product_names: list[str],
    top_k: int,
) -> list[str]:
    sc = scores[0].copy()
    sc[owned[0]] = -1.0
    order = np.argsort(-sc)
    chosen: list[str] = []
    for j in order:
        if owned[0, j]:
            continue
        chosen.append(product_names[j])
        if len(chosen) >= top_k:
            break
    return chosen


class RecommenderEngine:
    """Обёртка над артефактом обучения и таблицей признаков клиентов."""

    def __init__(self, model_path: Path, features_path: Path, default_top_k: int = 7):
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.default_top_k = int(default_top_k)
        self.artifact: dict[str, Any] | None = None
        self.features: pd.DataFrame | None = None
        self._id_index: dict[int, int] = {}

    @property
    def ready(self) -> bool:
        return self.artifact is not None and self.features is not None and len(self._id_index) > 0

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Нет модели: {self.model_path}")
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Нет features store: {self.features_path}. "
                "Сначала: python -m scripts.export_serving_features"
            )

        self.artifact = joblib.load(self.model_path)
        feat = pd.read_parquet(self.features_path)
        if ID_COL not in feat.columns:
            raise KeyError(f"В {self.features_path} нет колонки {ID_COL}")

        if DATE_COL in feat.columns:
            feat = feat.sort_values(DATE_COL).groupby(ID_COL, as_index=False).tail(1)
        else:
            feat = feat.drop_duplicates(subset=[ID_COL], keep="last")

        feat = feat.reset_index(drop=True)
        self.features = feat
        self._id_index = {int(cid): i for i, cid in enumerate(feat[ID_COL].to_numpy())}

    def has_client(self, ncodpers: int) -> bool:
        return int(ncodpers) in self._id_index

    def n_clients(self) -> int:
        return len(self._id_index)

    def recommend(self, ncodpers: int, top_k: int | None = None) -> list[dict[str, float | str]]:
        if not self.ready or self.artifact is None or self.features is None:
            raise RuntimeError("Движок не загружен")

        idx = self._id_index.get(int(ncodpers))
        if idx is None:
            raise KeyError(ncodpers)

        k = int(top_k or self.default_top_k)
        products = list(self.artifact.get("products") or PRODUCT_COLS)
        k = max(1, min(k, len(products)))

        row = self.features.iloc[[idx]]
        feat_cols = self.artifact["feature_columns"]
        models = self.artifact["models"]
        popularity = self.artifact["popularity"]
        train_positives = self.artifact.get("train_positives") or {}
        bayes_cfg = self.artifact.get("bayes_shrink") or {}
        bayes_prior = None
        if bayes_cfg.get("enabled") and bayes_cfg.get("prior_strength") is not None:
            bayes_prior = float(bayes_cfg["prior_strength"])

        scores = _predict_scores(
            models,
            row,
            feat_cols,
            popularity,
            products,
            train_positives=train_positives,
            bayes_prior=bayes_prior,
        )
        owned = _owned_matrix(row, products)
        ranked = _rank_fast(owned, scores, products, k)
        score_map = {products[j]: float(scores[0, j]) for j in range(len(products))}
        return [{"product": p, "score": round(score_map[p], 6)} for p in ranked]
