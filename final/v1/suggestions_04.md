# Журнал экспериментов (этап 4)

> Не затирать прошлые записи — **дописывать** новые в конец.
>
> Сводка: [`models/experiments_registry.json`](models/experiments_registry.json)  
> Текущий лучший: **v7** (hybrid + 100k + early stopping), MAP@7 = **0.02566** → [`models/model.bin`](models/model.bin)

Общий пайплайн (признаки / split / seed не менялись между прогонами):

```bash
python -m src.models.train --config configs/train.yaml --skip-mlflow
```

Ключевые файлы: [`src/models/train.py`](src/models/train.py), [`src/features/build.py`](src/features/build.py), [`configs/train.yaml`](configs/train.yaml).

---

## Сводная таблица

**Блок A — сэмпл 60k/80k** (valid 320k; сравнимы между собой):

| Модель | MAP@7 | Precision@7 | Recall@7 | Статус |
|--------|------:|------------:|---------:|--------|
| Popularity baseline | 0.02099 | 0.00569 | 0.03066 | — |
| v1 LightGBM (min_pos=20) | 0.02549 | 0.00573 | 0.03099 | база |
| v2 SPW `neg/pos`, cap=50 | 0.02313 | 0.00565 | 0.03050 | хуже v1 (−9.25%) |
| v3 SPW `sqrt(neg/pos)`, cap=50 | 0.02311 | 0.00568 | 0.03071 | хуже v1 (−9.32%) |
| v4 portfolio lags `[1,2]` | 0.02541 | 0.00575 | 0.03109 | чуть хуже v1 (−0.32%) |
| v5 hybrid min_pos=100 | 0.02556 | 0.00572 | 0.03090 | лучший на 60k (+0.30% к v1) |

**Блок B — сэмпл 100k/133k** (valid 533k; абсолютный MAP не сравнивать с блоком A):

| Модель | MAP@7 | vs baseline | Статус |
|--------|------:|------------:|--------|
| Popularity baseline | 0.02072 | — | — |
| v6 hybrid + 100k | 0.02553 | +23.2% | база 100k |
| **v7 + early stopping** | **0.02566** | **+23.8%** | **current** (+0.51% к v6) |
| bayes shrink m=100 | 0.02566 | +23.8% | ≈ v7, не берём |
| time weights hl=3 | 0.02556 | +23.3% | хуже v7 (−0.39%) |
| v8 bayes shrink m=100 | 0.02566 | +23.8% | ≈v7; не берём |

**Блок C — контроль user-split** (другие valid-строки; сравнивать lift, не абсолютный MAP с B):

| Модель | MAP@7 | vs baseline | Статус |
|--------|------:|------------:|--------|
| Popularity (user valid) | 0.02265 | — | — |
| user-split = v7 settings | 0.02770 | +22.3% | контроль; прод = time v7 |


---

## Общий код: `scale_pos_weight`

Реализовано в `src/models/train.py` и используется в экспериментах 1–2.

```python
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
```

Фрагмент обучения per-product (упрощённо):

```python
product_params = dict(lgb_params)
if use_scale_pos_weight:
    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)
    spw = _scale_pos_weight(n_pos, n_neg, spw_mode, scale_pos_weight_cap)
    product_params["scale_pos_weight"] = spw

model = lgb.LGBMClassifier(**product_params)
model.fit(x_train[tr_mask], y_tr)
```

Параметры читаются из конфига:

```python
use_spw = bool(model_cfg.get("use_scale_pos_weight", False))
spw_cap = float(model_cfg.get("scale_pos_weight_cap", 50.0))
spw_mode = str(model_cfg.get("scale_pos_weight_mode", "ratio"))
```

Гиперпараметры LightGBM (общие для v1–v3):

```yaml
model:
  params:
    n_estimators: 150
    learning_rate: 0.05
    num_leaves: 63
    subsample: 0.8
    colsample_bytree: 0.8
  top_k: 7
```

---

## Эксперимент 1: SPW = `neg/pos` (cap=50)

| Поле | Значение |
|------|----------|
| Run | `bank-rec-train-002-spw` |
| Артефакты | [`metrics_compare_spw.json`](models/metrics_compare_spw.json), [`metrics_spw_ratio.json`](models/metrics_spw_ratio.json) |

### Идея

Редкие открытия `0→1` тонут среди нулей. Для каждого продукта:

\[
\mathrm{scale\_pos\_weight} = \min(\mathrm{neg}/\mathrm{pos},\; 50)
\]

чтобы сильнее штрафовать пропуск позитива. Признаки, split, seed и сэмпл — как в v1 (без подглядывания в будущее).

### Конфиг эксперимента

```yaml
model:
  use_scale_pos_weight: true
  scale_pos_weight_mode: ratio   # neg/pos
  scale_pos_weight_cap: 50

mlflow:
  train_run_name: bank-rec-train-002-spw
```

### Защита от переобучения

- тот же time-based valid;
- потолок веса `cap=50`;
- те же гиперпараметры и `random_seed=42`.

### Результаты (valid)

| | MAP@7 | Precision@7 | Recall@7 |
|--|------:|------------:|---------:|
| baseline | 0.02099 | — | — |
| v1 | 0.02549 | 0.00573 | 0.03099 |
| v2 + SPW ratio | 0.02313 | 0.00565 | 0.03050 |
| delta v2 vs v1 | **−0.00236 (−9.25%)** | ↓ | ↓ |

### Наблюдение

У части продуктов AUC на valid просел (особенно средне-редких): вес 50 оказался слишком агрессивным — почти везде упёрлись в cap, модель стала «кричать» позитив и хуже ранжировать top-7.

### Вывод

Шаг **не** улучшил главную метрику. Лучшей остаётся **v1** (без SPW).

---

## Эксперимент 2: мягкий SPW = `sqrt(neg/pos)` (cap=50)

| Поле | Значение |
|------|----------|
| Run | `bank-rec-train-003-spw-sqrt` |
| Артефакты | [`metrics_compare_spw_sqrt.json`](models/metrics_compare_spw_sqrt.json), [`metrics_spw_sqrt.json`](models/metrics_spw_sqrt.json) |

### Идея

Тот же дисбаланс, но вес мягче:

\[
\mathrm{scale\_pos\_weight} = \min(\sqrt{\mathrm{neg}/\mathrm{pos}},\; 50)
\]

Для частых продуктов веса ~7–20; для редких всё ещё упираются в `cap=50`.

### Конфиг эксперимента

```yaml
model:
  use_scale_pos_weight: true
  scale_pos_weight_mode: sqrt
  scale_pos_weight_cap: 50

mlflow:
  train_run_name: bank-rec-train-003-spw-sqrt
```

### Защита от переобучения

- time-based valid;
- `sqrt` вместо сырого `neg/pos`;
- те же гиперпараметры и `seed=42`.

### Результаты (valid)

| | MAP@7 | Precision@7 | Recall@7 |
|--|------:|------------:|---------:|
| baseline | 0.02099 | — | — |
| v1 | 0.02549 | 0.00573 | 0.03099 |
| v2 SPW ratio | 0.02313 | — | — |
| v3 SPW sqrt | 0.02311 | 0.00568 | 0.03071 |
| delta v3 vs v1 | **−0.00238 (−9.32%)** | ↓ | ↓ |

### Наблюдение

`sqrt` снизил вес у популярных продуктов, но MAP@7 почти как у жёсткого SPW и всё ещё хуже v1. Ранжирование top-7 снова страдает от перевеса позитива. У редких продуктов AUC просел (`ctju` / `plan` ~0.5).

### Вывод

Мягкий SPW тоже **не** улучшил главную метрику. Лучшая — **v1** без SPW. Ветка SPW закрыта:

```yaml
model:
  use_scale_pos_weight: false
  scale_pos_weight_mode: sqrt
  scale_pos_weight_cap: 50
```

---

## Эксперимент 3: лаги портфеля (t−1 / t−2)

| Поле | Значение |
|------|----------|
| Run | `bank-rec-train-004-lags` |
| Артефакты | [`metrics_compare_lags.json`](models/metrics_compare_lags.json), [`metrics_lags.json`](models/metrics_lags.json), [`metrics_bank-rec-train-004-lags.json`](models/metrics_bank-rec-train-004-lags.json) |

### Идея

Добавить историю владения продуктами **строго до** месяца `t` (без утечки в t+1):

- `lag1_has_*`, `lag2_has_*` — портфель на t−1 / t−2 из `months_slim` (полный месяц, не из сэмпла);
- `n_products_lag1/2`, `lag*_missing`;
- `opened_lag1_*`, `n_opened_lag1`, `delta_n_products_lag1` — открытия между t−1 и t.

Loss / SPW / гиперпараметры / split / seed — как в v1. Признаков: 43 → **121**.

### Конфиг эксперимента

```yaml
features:
  portfolio_lags: [1, 2]

model:
  use_scale_pos_weight: false

mlflow:
  train_run_name: bank-rec-train-004-lags
```

Прогон:

```bash
python -m src.features.build --config configs/train.yaml
python -m src.models.train --config configs/train.yaml --skip-mlflow
```

### Код (фрагмент `src/features/build.py`)

```python
def attach_portfolio_lags(df, slim_dir, lags):
    """Джойн портфеля на t−k из months_slim (полный месяц, не из сэмпла)."""
    months = _slim_months(slim_dir)
    month_pos = {m: i for i, m in enumerate(months)}
    out = df.copy()
    # для каждого lag и каждой fecha_dato:
    #   src = months[idx - lag]
    #   merge products → lag{k}_has_{product}
    # при lag==1 дополнительно:
    #   opened_lag1_p = (lag1_has_p == 0) & (has_p == 1)
    ...
```

Ключевой момент: лаги берутся из **полного** slim-месяца по `ncodpers`, а не `shift` внутри сэмпла (иначе дыры из‑за `clients_per_month_*`).

### Защита от переобучения

- только прошлое относительно `fecha_dato`;
- тот же time-based valid;
- без изменения loss / без тюнинга по valid;
- `random_seed=42`, те же `n_estimators` / `num_leaves`.

### Результаты (valid)

| | MAP@7 | Precision@7 | Recall@7 | n_features |
|--|------:|------------:|---------:|-----------:|
| v1 | **0.02549** | 0.00573 | 0.03099 | 43 |
| v4 + lags | 0.02541 | 0.00575 | 0.03109 | 121 |
| delta v4 vs v1 | **−0.00008 (−0.32%)** | ↑ | ↑ | +78 |

### Наблюдение

Precision/Recall@7 слегка выросли, но главная метрика **MAP@7** чуть просела. Лаги не дали устойчивого выигрыша в ранжировании top‑7 на этом сэмпле: сигнал «что уже есть / что открыли недавно» в основном уже покрыт `has_*` на t. Часть AUC по средне-редким продуктам выросла (например `ctma`), но на MAP@7 это почти не отразилось.

### Вывод

Лаги **не** улучшили MAP@7 → на тот момент лучшей оставалась **v1**. В конфиге portfolio_lags: [], код лагов в uild.py сохранён для повторных опытов.

---

## Эксперимент 4: hybrid для редких (min_positives_train=100)

| Поле | Значение |
|------|----------|
| Run | ank-rec-train-005-hybrid-min100 |
| Артефакты | [metrics_compare_hybrid.json](models/metrics_compare_hybrid.json), [metrics_hybrid_min100.json](models/metrics_hybrid_min100.json), [model_v5_hybrid.bin](models/model_v5_hybrid.bin) |

### Идея

Модели на ультра-редких продуктах шумят в топ‑7 (низкий/случайный AUC). Hybrid:

- если позитивов на train < min_positives_train → **не учим** LightGBM, score = **popularity**;
- иначе — обычный LightGBM как в v1.

В v1 порог был захардкожен как 20. Здесь поднимаем до **100**.

Дополнительно ушли в popularity (сверх v1): ctju (22), deme (23), plan (48). Всего skip = 9 продуктов.

### Конфиг

`yaml
model:
  use_scale_pos_weight: false
  min_positives_train: 100   # было фактически 20

mlflow:
  train_run_name: bank-rec-train-005-hybrid-min100
`

### Код (src/models/train.py)

`python
n_pos = int(y_tr.sum())
if n_pos < min_pos:
    models[p] = None
    aucs[p] = float("nan")
    skipped.append(p)
    print(f"  skip {p}: positives={n_pos} < {min_pos} → popularity")
    continue
# ...
# в predict_scores:
if model is None:
    scores[:, j] = popularity.get(p, 0.0)
`

### Защита от переобучения

- тот же time-based valid / seed / гиперпараметры / признаки (43);
- порог выбран заранее (100), без перебора по valid;
- меньше «шумных» моделей → меньше риск переобучения на редких позитивах.

### Результаты (valid)

| | MAP@7 | Precision@7 | Recall@7 |
|--|------:|------------:|---------:|
| v1 (min_pos=20) | 0.02549 | 0.00573 | 0.03099 |
| **v5 (min_pos=100)** | **0.02556** | 0.00572 | 0.03090 |
| delta v5 vs v1 | **+0.00008 (+0.30%)** | ↓ | ↓ |

### Наблюдение

Прирост MAP@7 небольшой, но по смыслу стабильный: убрали слабые модели (ctju/plan с AUC ≪ 0.5 в прошлых прогонах), редкие слоты в ранжировании отдали popularity. P@7/R@7 чуть ниже — ожидаемо при более осторожном покрытии хвоста.

### Вывод

Hybrid **улучшил** главную метрику → новый лучший = **v5**. В конфиге оставляем min_positives_train: 100, models/model.bin = v5.

---

---

## Эксперимент 5: больший сэмпл (60k→100k / 80k→133k)

| Поле | Значение |
|------|----------|
| Run | `bank-rec-train-006-sample100k` |
| Артефакты | [`metrics_compare_sample100k.json`](models/metrics_compare_sample100k.json), [`metrics_sample100k.json`](models/metrics_sample100k.json), [`model_v6_sample100k.bin`](models/model_v6_sample100k.bin) |

### Идея

Уменьшить искажения от сэмплирования: больше клиентов на месяц при той же пропорции train/valid.

| | Было (v1–v5) | Стало (v6) |
|--|--:|--:|
| clients/month train | 60 000 | **100 000** |
| clients/month valid | 80 000 | **133 333** (=100k×80/60) |
| строк train | 720 000 | **1 200 000** |
| строк valid | 320 000 | **533 332** |

Модель: тот же hybrid `min_positives_train=100`, без SPW, без лагов, те же гиперпараметры и seed.

### Конфиг

```yaml
data:
  clients_per_month_train: 100000
  clients_per_month_valid: 133333

model:
  min_positives_train: 100
  use_scale_pos_weight: false

mlflow:
  train_run_name: bank-rec-train-006-sample100k
```

Прогон:

```bash
python -m src.data.prepare --config configs/train.yaml
python -m src.features.build --config configs/train.yaml
python -m src.models.train --config configs/train.yaml --skip-mlflow
```

### Защита от переобучения

- тот же time-based split;
- порог hybrid и гиперпараметры **не** подбирались на новом valid;
- сравниваем прежде всего **lift к baseline на том же сэмпле**, а не абсолютный MAP с блоком A.

### Результаты

| | baseline MAP@7 | model MAP@7 | lift |
|--|---------------:|------------:|-----:|
| v5 (60k/80k) | 0.02099 | 0.02556 | +21.8% |
| **v6 (100k/133k)** | 0.02072 | **0.02553** | **+23.2%** |

Skip → popularity: те же 9 редких продуктов (в т.ч. `plan` с 64 pos < 100).

### Наблюдение

Абсолютный MAP@7 почти как у v5, но valid другой — цифры не сравнимы напрямую. На большем сэмпле **преимущество над popularity сохраняется** (lift +23% vs +22%). Позитивов на частых продуктах больше (`recibo` 9k→15k) — оценки стабильнее.

### Вывод

Больший сэмпл **оставляем** как рабочий. Текущий `model.bin` = **v6**. Конфиг: 100k / 133333.

---

---

## Эксперимент 6: early stopping по holdout внутри train

| Поле | Значение |
|------|----------|
| Run | `bank-rec-train-007-early-stopping` |
| Артефакты | [`metrics_compare_early_stopping.json`](models/metrics_compare_early_stopping.json), [`metrics_early_stopping.json`](models/metrics_early_stopping.json), [`model_v7_early_stopping.bin`](models/model_v7_early_stopping.bin) |

### Идея

Не фиксировать `n_estimators=150` для всех продуктов. Остановить бустинг, когда logloss на **последнем месяце train** (2015-12) перестаёт улучшаться. Final valid (2016-01…04) **не** участвует в выборе числа деревьев.

### Конфиг

```yaml
model:
  params:
    n_estimators: 300   # потолок
  min_positives_train: 100
  early_stopping:
    enabled: true
    holdout_months: 1
    patience: 20
```

### Код (идея)

```python
fit_mask, es_mask, es_months = _time_holdout_masks(train, holdout_months=1)
model.fit(
    X_fit, y_fit,
    eval_set=[(X_es, y_es)],
    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)],
)
```

### Защита от переобучения

- ES только на куске train, не на final valid;
- hybrid / сэмпл / seed без изменений;
- patience и holdout заданы заранее, без сетки по valid.

### Результаты (тот же valid 533k)

| | MAP@7 | lift vs baseline | примечание |
|--|------:|-----------------:|------------|
| v6 (150 деревьев) | 0.02553 | +23.2% | |
| **v7 + ES** | **0.02566** | **+23.8%** | best_iter типично 2…183 |

delta MAP@7 vs v6: **+0.00013 (+0.51%)**.

### Наблюдение

У разных продуктов разная «нужная» глубина: редкие/средние останавливаются рано (`ctpp`=2, `fond`=3), массовые берут больше (`recibo`=183, `cco`=110). Фиксированные 150 деревьев для всех — компромисс хуже.

### Вывод

Early stopping **оставляем**: новый лучший = **v7**. Конфиг с `early_stopping.enabled: true`.

---

## Эксперимент 7: сплит по пользователям (контроль к time-split)

Параллельный пайплайн — **не** заменяет time-based prepare/train.

| | |
|--|--|
| Run | `bank-rec-train-user-split-001` |
| Config | [`configs/train_user_split.yaml`](configs/train_user_split.yaml) |
| Prepare | `python -m src.data.prepare_user_split --config configs/train_user_split.yaml` |
| Данные | `data/processed/datasets_user_split/` |
| Артефакты | [`metrics_user_split.json`](models/metrics_user_split.json), [`model_user_split.bin`](models/model_user_split.bin), [`metrics_compare_user_split.json`](models/metrics_compare_user_split.json) |

### Идея

Те же настройки модели, что у **v7** (hybrid min_pos=100, early stopping, сэмпл 100k/133k на месяц). Отличие только в разбиении:

- **time (v7):** train ≤2015-12, valid 2016-01…04;
- **user:** месяцы 2015-01…2016-04 в обоих фолдах; клиенты (`ncodpers`) 80/20 без пересечения.

Ограничение: на user-split модель видит «будущие» месяцы у train-клиентов → возможен leakage временных паттернов. Это контроль, не продакшен-валидация.

### Команды

```bash
python -m src.data.prepare_user_split --config configs/train_user_split.yaml
python -m src.features.build --config configs/train_user_split.yaml
python -m src.models.train --config configs/train_user_split.yaml --skip-mlflow
```

### Результаты

| | split | n_train | n_valid | baseline MAP@7 | model MAP@7 | lift |
|--|-------|--------:|--------:|---------------:|------------:|-----:|
| **v7 time** | время | 1.2M | 533k | 0.02072 | **0.02566** | **+23.8%** |
| user-split | клиенты | 1.6M | 2.09M | 0.02265 | 0.02770 | +22.3% |

### Вывод

- Абсолютный MAP@7 на user-split выше, но **valid другой** (все месяцы + hold-out клиенты) — с time-v7 напрямую не сравнивать.
- Lift над popularity **сопоставим** и даже чуть ниже (+22.3% vs +23.8%) → модель не «раздута» только за счёт user-split.
- Основной критерий и прод-модель остаются **time-based v7** (`models/model.bin`). User-split артефакты лежат отдельно.

---

## Эксперимент 8: Beta–Binomial shrink к popularity

| | |
|--|--|
| Run | `bank-rec-train-008-bayes-shrink` |
| Config | [`configs/train_bayes_shrink.yaml`](configs/train_bayes_shrink.yaml) |
| Артефакты | [`metrics_bayes_shrink.json`](models/metrics_bayes_shrink.json), [`model_bayes_shrink.bin`](models/model_bayes_shrink.bin), [`metrics_compare_bayes_shrink.json`](models/metrics_compare_bayes_shrink.json) |

### Идея

Вместо жёсткого skip редких (`min_pos=100` → popularity) учим почти всё (`min_pos=1`) и на инференсе тянем скор к popularity:

```text
p = (n_pos · p̂ + m · pop) / (n_pos + m),   m = prior_strength = 100
```

- `n_pos ≪ m` → почти popularity (хвост не «кричит»);
- `n_pos ≫ m` → почти сырой LightGBM.

Данные / seed / ES / гиперпараметры — как у **v7** (тот же time-split сэмпл).

### Результаты

| | MAP@7 | lift | skip |
|--|------:|-----:|-----:|
| **v7 hybrid** | **0.02566** | +23.8% | 9 |
| bayes m=100 | 0.02566 | +23.8% | 1 (`ahor`) |

delta MAP@7 vs v7: **+0.000002 (~0%)**. P@7/R@7 чуть выше за счёт слабого сигнала хвоста, на MAP не видно.

У редких AUC на valid часто ≪ 0.5 (`ctju` 0.14, `plan` 0.27, `hip` 0.35) — без shrink они портили бы top-7; с `m=100` их вклад почти как popularity.

### Вывод

Bayes-shrink **не улучшает** MAP@7 относительно hybrid. По сути дублирует skip при `m≈100`. В прод оставляем **v7** (проще и прозрачнее). Код shrink в `train.py` / `serve.py` сохранён для повторных опытов (`prior_strength` 50/200).

---

## Эксперимент 9: экспоненциальные веса по времени

| | |
|--|--|
| Run | `bank-rec-train-009-time-weights` |
| Config | [`configs/train_time_weights.yaml`](configs/train_time_weights.yaml) |
| Артефакты | [`metrics_time_weights.json`](models/metrics_time_weights.json), [`model_time_weights.bin`](models/model_time_weights.bin), [`metrics_compare_time_weights.json`](models/metrics_compare_time_weights.json) |

### Идея

Свежие месяцы train важнее старых. Для каждой строки:

```text
w = 2 ** (−age_months / half_life),   half_life = 3, anchor = 2015-12-28
```

| Месяц | вес |
|-------|----:|
| 2015-01 | 0.079 |
| 2015-06 | 0.25 |
| 2015-09 | 0.50 |
| 2015-12 | 1.00 |

Передаётся в LightGBM как `sample_weight`. Остальное = **v7** (hybrid 100, ES, тот же сэмпл).

### Результаты

| | MAP@7 | lift vs baseline | vs v7 |
|--|------:|-----------------:|------:|
| **v7** (равные веса) | **0.02566** | +23.8% | — |
| time weights hl=3 | 0.02556 | +23.3% | **−0.39%** |

### Вывод

Экспоненциальные веса **чуть ухудшили** MAP@7. Либо дрейф 2015→2016 слабый на этом сэмпле, либо `half_life=3` слишком режет ранние месяцы (январь ~8% веса). В прод **не** берём; код/`time_weights` в конфиге оставлены для `half_life=6` или якоря на первый valid-месяц.

---

## Что пробовать следующим (один пункт за раз)

1. Порог hybrid `min_positives=50` или `200` — один контрольный прогон.
2. Узкие лаги (`n_opened_lag1` / `delta_n_products`) поверх v7.
3. (опц.) Time weights с `half_life=6` (мягче) — один прогон.
4. (опц.) Bayes `m=50` — чуть больше веса хвосту.
