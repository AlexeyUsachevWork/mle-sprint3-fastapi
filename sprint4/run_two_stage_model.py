import gc
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parent

print("Loading events...")
events = pd.read_parquet(ROOT / "events.par")
events = events.rename(columns={"book_id": "item_id"})
split_date = pd.to_datetime("2017-08-01").date()
events_test = events[events["started_at"] >= split_date]
label_split = pd.to_datetime("2017-09-15").date()
events_labels = events_test[events_test["started_at"] < label_split].copy()
events_test_2 = events_test[events_test["started_at"] >= label_split].copy()
del events, events_test
gc.collect()

print("Loading training candidates (full merge)...")
als = pd.read_parquet(
    ROOT / "candidates/training/als_recommendations.parquet",
    columns=["user_id", "item_id", "score"],
).rename(columns={"score": "als_score"})
cnt = pd.read_parquet(
    ROOT / "candidates/training/content_recommendations.parquet",
    columns=["user_id", "item_id", "score"],
).rename(columns={"score": "cnt_score"})
candidates = als.merge(cnt, on=["user_id", "item_id"], how="outer")
del als, cnt
gc.collect()
print(f"candidates: {len(candidates)}")

events_labels["target"] = 1
candidates = candidates.merge(
    events_labels[["user_id", "item_id", "target"]],
    on=["user_id", "item_id"],
    how="left",
)
candidates["target"] = candidates["target"].fillna(0).astype("int")
candidates_to_sample = candidates.groupby("user_id").filter(lambda x: x["target"].sum() > 0)
candidates_for_train = pd.concat(
    [
        candidates_to_sample.query("target == 1"),
        candidates_to_sample.query("target == 0")
        .groupby("user_id")
        .apply(lambda x: x.sample(4, random_state=0), include_groups=False),
    ]
)
del candidates, candidates_to_sample, events_labels
gc.collect()
print(f"candidates_for_train: {len(candidates_for_train)}")

features = ["als_score", "cnt_score"]
print("Training CatBoost...")
cb_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function="Logloss",
    verbose=100,
    random_seed=0,
)
cb_model.fit(
    Pool(candidates_for_train[features].fillna(0), candidates_for_train["target"])
)
del candidates_for_train
gc.collect()

print("Building candidates_to_rank...")
als_inf = pd.read_parquet(
    ROOT / "candidates/inference/als_recommendations.parquet",
    columns=["user_id", "item_id", "score"],
).rename(columns={"score": "als_score"})
cnt_inf = pd.read_parquet(
    ROOT / "candidates/inference/content_recommendations.parquet",
    columns=["user_id", "item_id", "score"],
).rename(columns={"score": "cnt_score"})
candidates_to_rank = als_inf.merge(cnt_inf, on=["user_id", "item_id"], how="outer")
del als_inf, cnt_inf
candidates_to_rank = candidates_to_rank[
    candidates_to_rank["user_id"].isin(events_test_2["user_id"].drop_duplicates())
]
del events_test_2
gc.collect()
print(f"candidates_to_rank: {len(candidates_to_rank)}")

print("Predicting...")
candidates_to_rank["cb_score"] = cb_model.predict_proba(
    Pool(candidates_to_rank[features].fillna(0))
)[:, 1]
candidates_to_rank = candidates_to_rank.sort_values(
    ["user_id", "cb_score"], ascending=[True, False]
)
max_recommendations_per_user = 100
candidates_to_rank["rank"] = candidates_to_rank.groupby("user_id").cumcount() + 1
final_recommendations = candidates_to_rank.query(
    "rank <= @max_recommendations_per_user"
)

out_path = ROOT / "final_recommendations.parquet"
final_recommendations[["user_id", "item_id", "rank"]].to_parquet(out_path, index=False)
print(f"Saved: {out_path}")

uid = 1049126
user_recs = final_recommendations[final_recommendations["user_id"] == uid].head(5)
print(user_recs[["item_id", "rank"]])
print(f"2nd item_id: {int(user_recs.iloc[1]['item_id'])}")
