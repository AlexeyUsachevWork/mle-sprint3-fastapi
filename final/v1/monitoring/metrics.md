# Метрики мониторинга сервиса рекомендаций

Метрики отдаются из кода FastAPI на `GET /metrics` и собираются Prometheus
(`monitoring/prometheus.yml`). Grafana читает Prometheus и показывает дашборд
`Bank Product Recommender API` (provisioning в `monitoring/grafana/`).

## Сервисные метрики (из кода)

| Метрика | Тип | Описание | Источник |
|---------|-----|----------|----------|
| `bank_rec_requests_total` | Counter | HTTP-запросы (`endpoint`, `status`) | `app/main.py` |
| `bank_rec_recommend_latency_seconds` | Histogram | Латентность `POST /recommend` | `app/main.py` |
| `bank_rec_recommend_size` | Histogram | Число продуктов в ответе | `app/main.py` |
| `bank_rec_errors_total` | Counter | Ошибки (`client_not_found`, `inference`, …) | `app/main.py` |

## Как смотреть

1. API: http://localhost:8000/metrics  
2. Prometheus: http://localhost:9090  
3. Grafana: http://localhost:3000 (дашборд в папке *Bank Recommender*)  
4. Health: http://localhost:8000/health  

Логин Grafana по умолчанию: `admin` / `admin` (или `GRAFANA_USER` / `GRAFANA_PASSWORD`).

## Офлайн-метрики модели

MAP@7 / Precision@7 / Recall@7 — в MLflow и `models/metrics.json` (этап обучения), не в Prometheus.
