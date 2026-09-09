# Проверка проекта с нуля (как ревьюер)

Чеклист по артефактам из `task.md` и текущему README.

---

## 0. Смоук по репозиторию (без запуска)

- [ ] Есть `README.md`: цели, стек, структура, быстрый старт, руководство (бизнес→метрики, EDA, обучение, сервис, мониторинг)
- [ ] Есть `requirements.txt`, `random_seed` в конфиге
- [ ] `.gitignore` режет `.csv`, кэши, `.env*`, тяжёлые данные/модели
- [ ] Ноутбуки: `notebooks/01_eda.ipynb`, `notebooks/02_experiments.ipynb`
- [ ] Код пайплайна: `src/`, API: `app/`, Docker: `Dockerfile` + `docker-compose.yml`
- [ ] Журнал экспериментов: `suggestions_04.md` / метрики; описание метрик сервиса: `monitoring/metrics.md`

---

## 1. Окружение

```bash
git clone <repo>
cd mle-pr-final
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env.local   # при необходимости поправить порты/пароли
```

Нужны: **Docker Desktop** (демон запущен), `curl`, `python`.

---

## 2. Данные

```bash
bash scripts/download_dataset.sh
# ожидание: data/raw/train_ver2.csv
```

Проверить, что CSV **не** в git.

---

## 3. EDA (артефакт 1)

- [ ] Открыть / прогнать `notebooks/01_eda.ipynb` (или хотя бы outputs + выводы)
- [ ] В README §EDA: таргет `0→1`, редкость открытий, time-split, план очистки
- [ ] Опционально: артефакты в `data/processed/eda/` после прогона

---

## 4. MLflow (артефакт 2)

```bash
docker compose --profile mlflow up -d mlflow
# UI: http://127.0.0.1:5000
```

- [ ] UI открывается
- [ ] Опционально: `python scripts/log_eda_mlflow.py` или логирование из ноутбука — run с артефактами EDA

---

## 5. Пайплайн обучения (артефакты 4 + 8)

Полный путь (долго) **или** проверка готовых `models/model.bin` + `models/metrics.json`, если автор приложил/восстановил через DVC:

```bash
python -m src.data.prepare --config configs/train.yaml
python -m src.features.build --config configs/train.yaml
python -m src.models.train --config configs/train.yaml --skip-mlflow
# с MLflow: без --skip-mlflow, tracking_uri = http://127.0.0.1:5000
```

Проверить:

- [ ] Time-split: train ≤2015-12, valid 2016-01…04
- [ ] Метрики: MAP@7 / P@7 / R@7 vs popularity baseline
- [ ] Есть `models/model.bin`, `models/metrics.json`
- [ ] В `suggestions_04.md` / `02_experiments.ipynb` — что пробовали и что оставили (v7)
- [ ] В MLflow (если не skip) — params/metrics/artifacts run’а обучения

---

## 6. Сервис + мониторинг (артефакты 5–6)

```bash
python -m scripts.export_serving_features
# → data/serving/clients_features.parquet

docker compose up -d --build api prometheus grafana
# или: bash scripts/start_service.sh  /  ./scripts/start_service.ps1
```

Проверки:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d "{\"ncodpers\": <id_из_store>, \"top_k\": 7}"
# неизвестный id → 404
curl -s http://127.0.0.1:8000/metrics | head
```

- [ ] `/docs` — контракт API
- [ ] Prometheus http://127.0.0.1:9090 — scrape API
- [ ] Grafana http://127.0.0.1:3000 (`admin` / пароль из `.env`) — дашборд
- [ ] `monitoring/metrics.md` описывает метрики из кода

Полный стек разом:

```bash
docker compose --profile mlflow up -d --build
```

---

## 7. Документация и воспроизводимость (артефакты 7–8)

- [ ] README достаточен, чтобы повторить шаги без «магии»
- [ ] Фиксированы seed и зависимости
- [ ] Понятно, какая модель в проде (`models/model.bin` = v7) и почему

---

## Минимальный путь (если нет времени на полный train)

1. clone + venv + requirements
2. данные (download) **или** готовые `models/` + `data/serving/` от автора
3. `docker compose --profile mlflow up -d mlflow`
4. `export_serving_features` (если нет parquet)
5. `docker compose up -d --build api prometheus grafana`
6. health / recommend / metrics / Grafana
7. Прочитать `01_eda`, `02_experiments`, `suggestions_04.md`, README §1–5

Без данных и без готовой модели полный прогон **не** сойдётся на шаге API — это нормально; ревьюеру нужны либо train с нуля, либо приложенные артефакты serving/model (не в git — уточнить у автора / DVC).

---

## Что считать «проект принят»

| Пункт task.md | Как убедиться |
|---------------|---------------|
| EDA notebook | Есть + выводы |
| MLflow | Docker UI + (желательно) run |
| Метрики/постановка | README |
| Модель + эксперименты | `model.bin` + notebook/журнал |
| FastAPI в Docker | `/recommend` работает |
| Мониторинг | `/metrics` + Prometheus/Grafana + `metrics.md` |
| README + requirements | воспроизводимый старт |
