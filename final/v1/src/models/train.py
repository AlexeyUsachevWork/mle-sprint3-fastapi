"""Обучение рекомендателя: baseline + LightGBM по продуктам, метрики@K, MLflow.

Использование:
    python -m src.models.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import DATE_COL, ID_COL, PRODUCT_COLS  # noqa: E402
from src.models.metrics import evaluate_ranking  # noqa: E402
from src.tracking import get_or_create_experiment, setup_mlflow  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _feature_matrix(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    return df[feat_cols].to_numpy(dtype=np.float32)


def _owned_mask(df: pd.DataFrame, product: str) -> np.ndarray:
    col = f"has_{product}"
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df[col].to_numpy() == 1


def _time_holdout_masks(train: pd.DataFrame, holdout_months: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Последние holdout_months месяцев train → ES; остальное → fit. Final valid не трогаем."""
    dates = train[DATE_COL].astype("string").str.slice(0, 10)
    months = sorted(dates.unique().tolist())
    if holdout_months <= 0 or len(months) <= holdout_months:
        return np.ones(len(train), dtype=bool), np.zeros(len(train), dtype=bool), []
    es_months = months[-holdout_months:]
    es_mask = dates.isin(es_months).to_numpy()
    fit_mask = ~es_mask
    return fit_mask, es_mask, list(es_months)

def popularity_baseline_scores(train: pd.DataFrame) -> dict[str, float]:
    """Доля открытий продукта на train (среди тех, у кого его ещё не было)."""
    scores = {}
    for p in PRODUCT_COLS:
        tcol = f"target_{p}"
        hcol = f"has_{p}"
        if tcol not in train.columns:
            scores[p] = 0.0
            continue
        eligible = train[train[hcol] == 0] if hcol in train.columns else train
        scores[p] = float(eligible[tcol].mean()) if len(eligible) else 0.0
    return scores


def bayes_shrink_scores(
    p_hat: np.ndarray,
    pop: float,
    n_pos: int,
    prior_strength: float,
) -> np.ndarray:
    """Beta–Binomial shrink к popularity: p = (n_pos·p̂ + m·pop) / (n_pos + m).

    При n_pos=0 или m→∞ → pop; при n_pos ≫ m → p̂.
    """
    m = float(prior_strength)
    n = float(max(int(n_pos), 0))
    if m <= 0:
        return p_hat.astype(np.float32, copy=False)
    if n <= 0:
        return np.full_like(p_hat, float(pop), dtype=np.float32)
    return ((n * p_hat + m * float(pop)) / (n + m)).astype(np.float32)


def rank_from_score_dict(
    df: pd.DataFrame,
    score_lookup: dict[str, float] | None,
    model_scores: dict[str, np.ndarray] | None,
    top_k: int,
) -> list[list[str]]:
    preds: list[list[str]] = []
    n = len(df)
    for i in range(n):
        items: list[tuple[str, float]] = []
        for p in PRODUCT_COLS:
            if _owned_mask(df.iloc[[i]], p)[0]:
                continue
            if model_scores is not None:
                sc = float(model_scores[p][i])
            else:
                assert score_lookup is not None
                sc = float(score_lookup.get(p, 0.0))
            items.append((p, sc))
        items.sort(key=lambda x: x[1], reverse=True)
        preds.append([p for p, _ in items[:top_k]])
    return preds


def rank_fast(
    owned: np.ndarray,
    scores: np.ndarray,
    product_names: list[str],
    top_k: int,
) -> list[list[str]]:
    """owned: (n, p) bool уже есть; scores: (n, p)."""
    preds: list[list[str]] = []
    n, pcount = scores.shape
    for i in range(n):
        sc = scores[i].copy()
        sc[owned[i]] = -1.0  # уже есть — в низ списка
        order = np.argsort(-sc)
        chosen = []
        for j in order:
            if owned[i, j]:
                continue
            chosen.append(product_names[j])
            if len(chosen) >= top_k:
                break
        preds.append(chosen)
    return preds


def true_product_sets(df: pd.DataFrame) -> list[set[str]]:
    target_cols = [f"target_{p}" for p in PRODUCT_COLS]
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Нет колонок таргета: {missing[:3]}...")
    mat = df[target_cols].fillna(0).to_numpy(dtype=np.int8)
    out: list[set[str]] = []
    for i in range(mat.shape[0]):
        idx = np.flatnonzero(mat[i] == 1)
        out.append({PRODUCT_COLS[j] for j in idx})
    return out


def _scale_pos_weight(
    n_pos: float,
    n_neg: float,
    mode: str,
    cap: float,
) -> float:
    """Вес позитивного класса. mode: ratio = neg/pos, sqrt = sqrt(neg/pos)."""
    ratio = n_neg / max(n_pos, 1.0)
    if mode == "sqrt":
        spw = float(ratio**0.5)
    else:
        spw = float(ratio)
    return float(min(spw, cap))


def exponential_time_weights(
    dates: pd.Series,
    half_life_months: float,
    anchor: str | None = None,
) -> np.ndarray:
    """Веса строк: экспоненциально растут к свежим месяцам.

    w = 2 ** (-age_months / half_life), age = число месяцев до якоря
    (последний месяц в dates или явный anchor, напр. train_end).
    Самый свежий месяц → вес 1.0.
    """
    d = dates.astype("string").str.slice(0, 10)
    months = sorted(d.unique().tolist())
    if not months:
        return np.ones(len(dates), dtype=np.float32)
    anchor_m = str(anchor) if anchor else months[-1]
    if anchor_m not in months:
        # якорь позже всех train-месяцев — считаем age от последнего train
        months_for_idx = months + ([anchor_m] if anchor_m > months[-1] else [])
        if anchor_m not in months_for_idx:
            months_for_idx = sorted(set(months) | {anchor_m})
    else:
        months_for_idx = months
    idx = {m: i for i, m in enumerate(months_for_idx)}
    anchor_i = idx.get(anchor_m, len(months) - 1)
    hl = max(float(half_life_months), 1e-6)
    age = d.map(lambda m: float(anchor_i - idx.get(m, anchor_i))).to_numpy(dtype=np.float64)
    age = np.clip(age, 0.0, None)
    return np.power(2.0, -age / hl).astype(np.float32)


def train_lgbm_models(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feat_cols: list[str],
    params: dict,
    seed: int,
    use_scale_pos_weight: bool = False,
    scale_pos_weight_cap: float = 50.0,
    scale_pos_weight_mode: str = "ratio",
    min_positives_train: int = 20,
    early_stopping: bool = False,
    es_holdout_months: int = 1,
    es_patience: int = 20,
    time_weights: np.ndarray | None = None,
) -> tuple[
    dict[str, lgb.LGBMClassifier],
    dict[str, float],
    list[str],
    dict[str, int | None],
    dict[str, int],
]:
    models: dict[str, lgb.LGBMClassifier] = {}
    aucs: dict[str, float] = {}
    skipped: list[str] = []
    best_iters: dict[str, int | None] = {}
    train_positives: dict[str, int] = {}
    x_all = _feature_matrix(train, feat_cols)
    x_valid = _feature_matrix(valid, feat_cols)
    w_all = time_weights if time_weights is not None else np.ones(len(train), dtype=np.float32)

    fit_time_mask, es_time_mask, es_months = _time_holdout_masks(
        train, es_holdout_months if early_stopping else 0
    )
    if early_stopping:
        print(f"  early_stopping: holdout months={es_months}, patience={es_patience}")

    lgb_params = {
        "n_estimators": int(params.get("n_estimators", 200)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "num_leaves": int(params.get("num_leaves", 63)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    spw_mode = (scale_pos_weight_mode or "ratio").lower()
    min_pos = int(min_positives_train)

    for p in PRODUCT_COLS:
        tcol = f"target_{p}"
        hcol = f"has_{p}"
        eligible = train[hcol].to_numpy() == 0 if hcol in train.columns else np.ones(len(train), bool)
        va_mask = valid[hcol].to_numpy() == 0 if hcol in valid.columns else np.ones(len(valid), bool)
        y_eligible = train.loc[eligible, tcol].to_numpy()
        n_pos = int(y_eligible.sum())
        train_positives[p] = n_pos
        if n_pos < min_pos:
            models[p] = None  # type: ignore[assignment]
            aucs[p] = float("nan")
            best_iters[p] = None
            skipped.append(p)
            print(f"  skip {p}: positives={n_pos} < {min_pos} → popularity")
            continue

        fit_mask = eligible & fit_time_mask
        es_mask = eligible & es_time_mask
        y_fit = train.loc[fit_mask, tcol].to_numpy()
        y_va = valid.loc[va_mask, tcol].to_numpy()

        # ES только если в fit достаточно позитивов (не раздуваем редкие деревьями)
        es_min_pos = max(min_pos, 20)
        use_es = bool(early_stopping and es_mask.any() and int(y_fit.sum()) >= es_min_pos)
        if early_stopping and not use_es:
            fit_mask = eligible
            y_fit = y_eligible

        product_params = dict(lgb_params)
        if use_scale_pos_weight:
            n_pos_f = float(y_fit.sum())
            n_neg = float(len(y_fit) - n_pos_f)
            spw = _scale_pos_weight(n_pos_f, n_neg, spw_mode, scale_pos_weight_cap)
            product_params["scale_pos_weight"] = spw
        else:
            spw = None

        w_fit = w_all[fit_mask]
        model = lgb.LGBMClassifier(**product_params)
        if use_es:
            y_es = train.loc[es_mask, tcol].to_numpy()
            model.fit(
                x_all[fit_mask],
                y_fit,
                sample_weight=w_fit,
                eval_set=[(x_all[es_mask], y_es)],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=es_patience, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iter = int(getattr(model, "best_iteration_", 0) or 0)
            best_iters[p] = best_iter if best_iter > 0 else None
        else:
            model.fit(x_all[fit_mask], y_fit, sample_weight=w_fit)
            best_iters[p] = None

        proba = model.predict_proba(x_valid[va_mask])[:, 1]
        try:
            aucs[p] = float(roc_auc_score(y_va, proba)) if y_va.sum() > 0 and y_va.sum() < len(y_va) else float("nan")
        except ValueError:
            aucs[p] = float("nan")
        models[p] = model
        spw_txt = f", spw={spw:.1f}" if spw is not None else ""
        es_txt = f", best_iter={best_iters[p]}" if best_iters[p] is not None else ""
        if aucs[p] == aucs[p]:
            print(f"  {p}: pos_train={n_pos}, auc={aucs[p]:.4f}{spw_txt}{es_txt}")
        else:
            print(f"  {p}: pos_train={n_pos}, auc=nan{spw_txt}{es_txt}")

    return models, aucs, skipped, best_iters, train_positives



def predict_scores(
    models: dict[str, lgb.LGBMClassifier | None],
    df: pd.DataFrame,
    feat_cols: list[str],
    popularity: dict[str, float],
    train_positives: dict[str, int] | None = None,
    bayes_prior: float | None = None,
) -> np.ndarray:
    x = _feature_matrix(df, feat_cols)
    n = len(df)
    scores = np.zeros((n, len(PRODUCT_COLS)), dtype=np.float32)
    pos_map = train_positives or {}
    use_shrink = bayes_prior is not None and float(bayes_prior) > 0
    for j, p in enumerate(PRODUCT_COLS):
        pop = float(popularity.get(p, 0.0))
        model = models.get(p)
        if model is None:
            scores[:, j] = pop
            continue
        p_hat = model.predict_proba(x)[:, 1]
        if use_shrink:
            scores[:, j] = bayes_shrink_scores(p_hat, pop, int(pos_map.get(p, 0)), float(bayes_prior))
        else:
            scores[:, j] = p_hat
    return scores


def owned_matrix(df: pd.DataFrame) -> np.ndarray:
    mats = []
    for p in PRODUCT_COLS:
        col = f"has_{p}"
        if col in df.columns:
            mats.append((df[col].to_numpy() == 1))
        else:
            mats.append(np.zeros(len(df), dtype=bool))
    return np.vstack(mats).T


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение рекомендателя")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "configs" / "train.yaml")
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("random_seed", 42))
    model_cfg = cfg.get("model") or {}
    top_k = int(model_cfg.get("top_k", 7))
    params = model_cfg.get("params") or {}
    use_spw = bool(model_cfg.get("use_scale_pos_weight", False))
    spw_cap = float(model_cfg.get("scale_pos_weight_cap", 50.0))
    spw_mode = str(model_cfg.get("scale_pos_weight_mode", "ratio"))
    min_pos = int(model_cfg.get("min_positives_train", 20))
    es_cfg = model_cfg.get("early_stopping") or {}
    use_es = bool(es_cfg.get("enabled", False))
    es_holdout = int(es_cfg.get("holdout_months", 1))
    es_patience = int(es_cfg.get("patience", 20))
    bayes_cfg = model_cfg.get("bayes_shrink") or {}
    use_bayes = bool(bayes_cfg.get("enabled", False))
    bayes_prior = float(bayes_cfg.get("prior_strength", 100.0)) if use_bayes else None
    tw_cfg = model_cfg.get("time_weights") or {}
    use_tw = bool(tw_cfg.get("enabled", False))
    tw_half_life = float(tw_cfg.get("half_life_months", 3.0))
    tw_anchor = tw_cfg.get("anchor") or (cfg.get("split") or {}).get("train_end")
    paths = cfg.get("paths") or {}
    mlflow_cfg = cfg.get("mlflow") or {}

    data_cfg = cfg.get("data") or {}
    processed = ROOT_DIR / data_cfg.get("processed_dir", "data/processed")
    ds = processed / data_cfg.get("datasets_dir", "datasets")
    train = pd.read_parquet(ds / "train_features.parquet")
    valid = pd.read_parquet(ds / "valid_features.parquet")
    bundle = joblib.load(ds / "category_maps.joblib")
    feat_cols = bundle["feature_columns"]

    print("Baseline popularity…")
    popularity = popularity_baseline_scores(train)
    owned_va = owned_matrix(valid)
    true_va = true_product_sets(valid)

    pop_scores = np.tile(
        np.array([popularity.get(p, 0.0) for p in PRODUCT_COLS], dtype=np.float32),
        (len(valid), 1),
    )
    baseline_preds = rank_fast(owned_va, pop_scores, PRODUCT_COLS, top_k)
    baseline_metrics = evaluate_ranking(true_va, baseline_preds, top_k)
    print("Baseline:", baseline_metrics)

    sample_w: np.ndarray | None = None
    if use_tw:
        sample_w = exponential_time_weights(train[DATE_COL], tw_half_life, anchor=tw_anchor)
        # сводка весов по месяцам
        months = train[DATE_COL].astype("string").str.slice(0, 10)
        by_m = (
            pd.DataFrame({"m": months, "w": sample_w})
            .groupby("m", sort=True)["w"]
            .first()
        )
        print(
            f"  time_weights: half_life={tw_half_life}, anchor={tw_anchor}, "
            f"w_min={float(sample_w.min()):.4f}, w_max={float(sample_w.max()):.4f}"
        )
        print("  weights by month:", {k: round(float(v), 4) for k, v in by_m.items()})

    print(
        f"LightGBM per product (scale_pos_weight={use_spw}, mode={spw_mode}, "
        f"cap={spw_cap}, min_positives={min_pos}, early_stopping={use_es}, "
        f"bayes_shrink={use_bayes}, prior={bayes_prior}, time_weights={use_tw})…"
    )
    models, aucs, skipped, best_iters, train_positives = train_lgbm_models(
        train,
        valid,
        feat_cols,
        params,
        seed,
        use_scale_pos_weight=use_spw,
        scale_pos_weight_cap=spw_cap,
        scale_pos_weight_mode=spw_mode,
        min_positives_train=min_pos,
        early_stopping=use_es,
        es_holdout_months=es_holdout,
        es_patience=es_patience,
        time_weights=sample_w,
    )
    print(f"Skipped → popularity ({len(skipped)}): {skipped}")
    model_scores = predict_scores(
        models,
        valid,
        feat_cols,
        popularity,
        train_positives=train_positives,
        bayes_prior=bayes_prior,
    )
    model_preds = rank_fast(owned_va, model_scores, PRODUCT_COLS, top_k)
    model_metrics = evaluate_ranking(true_va, model_preds, top_k)
    print("LightGBM:", model_metrics)

    artifact = {
        "model_type": "lightgbm_per_product",
        "products": PRODUCT_COLS,
        "feature_columns": feat_cols,
        "models": models,
        "popularity": popularity,
        "train_positives": train_positives,
        "category_maps": bundle["maps"],
        "top_k": top_k,
        "metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "product_aucs": aucs,
        "skipped_products": skipped,
        "best_iterations": best_iters,
        "random_seed": seed,
        "lgbm_params": params,
        "use_scale_pos_weight": use_spw,
        "scale_pos_weight_cap": spw_cap,
        "scale_pos_weight_mode": spw_mode,
        "min_positives_train": min_pos,
        "bayes_shrink": {
            "enabled": use_bayes,
            "prior_strength": bayes_prior if use_bayes else None,
        },
        "time_weights": {
            "enabled": use_tw,
            "half_life_months": tw_half_life if use_tw else None,
            "anchor": tw_anchor if use_tw else None,
        },
        "early_stopping": {
            "enabled": use_es,
            "holdout_months": es_holdout,
            "patience": es_patience,
        },
    }

    model_path = ROOT_DIR / paths.get("model_path", "models/model.bin")
    metrics_path = ROOT_DIR / paths.get("metrics_path", "models/metrics.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    metrics_payload = {
        "baseline": baseline_metrics,
        "lightgbm": model_metrics,
        "product_aucs": {k: (None if v != v else v) for k, v in aucs.items()},
        "skipped_products": skipped,
        "best_iterations": best_iters,
        "train_positives": train_positives,
        "top_k": top_k,
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_features": len(feat_cols),
        "use_scale_pos_weight": use_spw,
        "scale_pos_weight_cap": spw_cap,
        "scale_pos_weight_mode": spw_mode,
        "min_positives_train": min_pos,
        "bayes_shrink": {
            "enabled": use_bayes,
            "prior_strength": bayes_prior if use_bayes else None,
        },
        "time_weights": {
            "enabled": use_tw,
            "half_life_months": tw_half_life if use_tw else None,
            "anchor": tw_anchor if use_tw else None,
        },
        "early_stopping": {
            "enabled": use_es,
            "holdout_months": es_holdout,
            "patience": es_patience,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model → {model_path}")
    print(f"Saved metrics → {metrics_path}")

    # Снимок прогона с уникальным именем — чтобы эксперименты не затирали друг друга
    run_name = str(mlflow_cfg.get("train_run_name", "bank-rec-train"))
    safe_run = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_name)
    metrics_snap = model_path.parent / f"metrics_{safe_run}.json"
    metrics_snap.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics snapshot → {metrics_snap}")

    if args.skip_mlflow:
        return

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    tracking_uri = setup_mlflow(mlflow_cfg.get("tracking_uri"))
    exp_name = mlflow_cfg.get("experiment_name", "bank-product-recommender")
    exp_id = get_or_create_experiment(exp_name)

    with mlflow.start_run(run_name=run_name, experiment_id=exp_id) as run:
        mlflow.set_tags({"stage": "train", "has_model": "true", "model_type": "lightgbm_per_product"})
        mlflow.log_params(
            {
                "top_k": top_k,
                "random_seed": seed,
                "n_estimators": params.get("n_estimators"),
                "learning_rate": params.get("learning_rate"),
                "num_leaves": params.get("num_leaves"),
                "n_features": len(feat_cols),
                "n_train": len(train),
                "n_valid": len(valid),
                "use_scale_pos_weight": use_spw,
                "scale_pos_weight_cap": spw_cap,
                "scale_pos_weight_mode": spw_mode,
                "min_positives_train": min_pos,
                "early_stopping": use_es,
                "es_holdout_months": es_holdout,
                "es_patience": es_patience,
                "bayes_shrink": use_bayes,
                "bayes_prior_strength": bayes_prior if use_bayes else 0,
            }
        )
        for prefix, metrics in ("baseline", baseline_metrics), ("model", model_metrics):
            for k, v in metrics.items():
                safe = k.replace("@", "_at_")
                mlflow.log_metric(f"{prefix}_{safe}", float(v))
        for p, auc in aucs.items():
            if auc == auc:
                mlflow.log_metric(f"auc_{p}", float(auc))

        mlflow.log_artifact(str(metrics_path), artifact_path="train")
        mlflow.log_artifact(str(model_path), artifact_path="train")
        mlflow.log_dict(metrics_payload, "train/metrics.json")
        print(f"MLflow run: {run_name} ({run.info.run_id}) @ {tracking_uri}")


if __name__ == "__main__":
    main()
