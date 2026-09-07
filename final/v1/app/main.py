"""FastAPI-сервис рекомендаций банковских продуктов.

Паттерны из sprint-3/4: lifespan-загрузка, Pydantic-контракт, Prometheus-метрики,
ошибки 404/500, Docker + bind-mount моделей и features store.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.schemas import HealthResponse, RecommendRequest, RecommendResponse
from src.models.serve import RecommenderEngine

ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_service_cfg() -> dict:
    cfg_path = Path(os.getenv("SERVICE_CONFIG", ROOT_DIR / "configs" / "service.yaml"))
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


CFG = _load_service_cfg()
MODEL_PATH = Path(os.getenv("MODEL_PATH", CFG.get("model_path", "models/model.bin")))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT_DIR / MODEL_PATH
FEATURES_PATH = Path(
    os.getenv("FEATURES_PATH", CFG.get("features_path", "data/serving/clients_features.parquet"))
)
if not FEATURES_PATH.is_absolute():
    FEATURES_PATH = ROOT_DIR / FEATURES_PATH
DEFAULT_TOP_K = int(os.getenv("TOP_K", CFG.get("top_k", 7)))

REQUESTS = Counter(
    "bank_rec_requests_total",
    "Суммарное число HTTP-запросов к API рекомендаций",
    ["endpoint", "status"],
)
RECOMMEND_LATENCY = Histogram(
    "bank_rec_recommend_latency_seconds",
    "Латентность POST /recommend",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
RECOMMEND_SIZE = Histogram(
    "bank_rec_recommend_size",
    "Число продуктов в ответе /recommend",
    buckets=(0, 1, 2, 3, 5, 7, 10, 15, 24),
)
ERRORS = Counter(
    "bank_rec_errors_total",
    "Ошибки API",
    ["endpoint", "error_type"],
)

engine = RecommenderEngine(MODEL_PATH, FEATURES_PATH, DEFAULT_TOP_K)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.load()
    yield


app = FastAPI(
    title="Bank Product Recommender",
    version="1.0.0",
    description="API рекомендаций банковских продуктов клиентам.",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    REQUESTS.labels(endpoint="root", status="ok").inc()
    return {
        "service": "bank-product-recommender",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "recommend": "POST /recommend",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    ok = engine.ready
    REQUESTS.labels(endpoint="health", status="ok" if ok else "degraded").inc()
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_loaded=ok,
        n_clients=engine.n_clients(),
    )


@app.get("/metrics", tags=["meta"])
def metrics() -> PlainTextResponse:
    REQUESTS.labels(endpoint="metrics", status="ok").inc()
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(body: RecommendRequest) -> RecommendResponse:
    t0 = time.perf_counter()
    if not engine.ready:
        ERRORS.labels(endpoint="recommend", error_type="model_not_loaded").inc()
        REQUESTS.labels(endpoint="recommend", status="500").inc()
        raise HTTPException(status_code=500, detail="Модель или features store не загружены")

    if not engine.has_client(body.ncodpers):
        ERRORS.labels(endpoint="recommend", error_type="client_not_found").inc()
        REQUESTS.labels(endpoint="recommend", status="404").inc()
        raise HTTPException(
            status_code=404,
            detail=f"Клиент ncodpers={body.ncodpers} не найден в features store",
        )

    try:
        items = engine.recommend(body.ncodpers, body.top_k)
    except Exception as exc:  # noqa: BLE001
        ERRORS.labels(endpoint="recommend", error_type="inference").inc()
        REQUESTS.labels(endpoint="recommend", status="500").inc()
        raise HTTPException(status_code=500, detail="Ошибка инференса") from exc

    elapsed = time.perf_counter() - t0
    RECOMMEND_LATENCY.observe(elapsed)
    RECOMMEND_SIZE.observe(len(items))
    REQUESTS.labels(endpoint="recommend", status="ok").inc()

    top_k = int(body.top_k or DEFAULT_TOP_K)
    return RecommendResponse(ncodpers=body.ncodpers, top_k=top_k, recommendations=items)
