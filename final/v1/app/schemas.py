"""Схемы запроса/ответа API рекомендаций."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    ncodpers: int = Field(..., description="ID клиента (как в датасете)")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=24,
        description="Сколько продуктов вернуть; по умолчанию из configs/service.yaml",
    )


class ProductScore(BaseModel):
    product: str
    score: float


class RecommendResponse(BaseModel):
    ncodpers: int
    top_k: int
    recommendations: list[ProductScore]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_clients: int
