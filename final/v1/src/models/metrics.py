"""Метрики ранжирования топ-K для рекомендаций продуктов."""

from __future__ import annotations

import numpy as np


def average_precision_at_k(y_true: set[str], y_pred: list[str], k: int) -> float:
    if not y_true:
        return 0.0
    hits = 0
    score = 0.0
    for i, p in enumerate(y_pred[:k], start=1):
        if p in y_true:
            hits += 1
            score += hits / i
    return score / min(len(y_true), k)


def precision_at_k(y_true: set[str], y_pred: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    pred = y_pred[:k]
    if not pred:
        return 0.0
    return len(y_true.intersection(pred)) / len(pred)


def recall_at_k(y_true: set[str], y_pred: list[str], k: int) -> float:
    if not y_true:
        return 0.0
    return len(y_true.intersection(y_pred[:k])) / len(y_true)


def evaluate_ranking(
    true_lists: list[set[str]],
    pred_lists: list[list[str]],
    k: int,
) -> dict[str, float]:
    aps, precs, recs = [], [], []
    for yt, yp in zip(true_lists, pred_lists):
        aps.append(average_precision_at_k(yt, yp, k))
        precs.append(precision_at_k(yt, yp, k))
        recs.append(recall_at_k(yt, yp, k))
    return {
        f"map@{k}": float(np.mean(aps)),
        f"precision@{k}": float(np.mean(precs)),
        f"recall@{k}": float(np.mean(recs)),
        "n_eval_rows": float(len(true_lists)),
        "share_with_positives": float(np.mean([1.0 if t else 0.0 for t in true_lists])),
    }
