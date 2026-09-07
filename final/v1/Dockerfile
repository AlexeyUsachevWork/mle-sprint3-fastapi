FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.bin \
    FEATURES_PATH=/app/data/serving/clients_features.parquet \
    SERVICE_CONFIG=/app/configs/service.yaml \
    TOP_K=7

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY app ./app
COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
