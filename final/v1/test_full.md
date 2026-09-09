# Полный путь с нуля (последовательность команд)

Корень репозитория. Windows: PowerShell; для `.sh` — Git Bash / WSL. Docker Desktop должен быть запущен.

```bash
# 1. Клон и окружение
git clone https://github.com/AlexeyUsachevWork/mle-pr-final.git
cd mle-pr-final
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Локальные настройки (порты/пароли Grafana при желании)
copy .env.example .env.local          # Linux: cp .env.example .env.local

# 3. Данные (~большой CSV с Яндекс.Диска)
bash scripts/download_dataset.sh      # → data/raw/train_ver2.csv

# 4. MLflow (Tracking UI)
docker compose --profile mlflow up -d mlflow
# открыть http://127.0.0.1:5000

# 5. EDA (ноутбук; тяжёлый полный прогон)
jupyter lab                          # открыть notebooks/01_eda.ipynb, Run All
# опционально залогировать EDA:
python scripts/log_eda_mlflow.py

# 6. Подготовка train/valid (time-split)
python -m src.data.prepare --config configs/train.yaml
# → data/processed/datasets/{train,valid}.parquet

# 7. Признаки
python -m src.features.build --config configs/train.yaml
# → *_features.parquet, category_maps.joblib

# 8. Обучение v7 (+ запись в MLflow; без UI добавьте --skip-mlflow)
python -m src.models.train --config configs/train.yaml
# → models/model.bin, models/metrics.json
# смотреть run в http://127.0.0.1:5000

# 9. Features store для API
python -m scripts.export_serving_features
# → data/serving/clients_features.parquet

# 10. Serving-стек: API + Prometheus + Grafana
docker compose up -d --build api prometheus grafana
# API http://127.0.0.1:8000/docs
# Prometheus http://127.0.0.1:9090
# Grafana http://127.0.0.1:3000  (admin / GRAFANA_PASSWORD из .env.local)

# 11. Смоук API
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"ncodpers\": 1379773, \"top_k\": 7}"
# Linux/Git Bash: одна строка с \, неизвестный id → 404
curl -s http://127.0.0.1:8000/metrics | more

# 12. (опционально) ноутбук экспериментов — обзор, не обязателен для сервиса
# notebooks/02_experiments.ipynb
```

Остановка:

```bash
docker compose --profile mlflow down
```

**Порядок важен:** данные → (MLflow) → EDA по желанию → prepare → features → train → export serving → Docker API/мониторинг. Без шагов 6–9 контейнер `api` не поднимется нормально (нет `model.bin` / parquet).
