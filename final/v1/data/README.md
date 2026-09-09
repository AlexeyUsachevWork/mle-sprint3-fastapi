# Данные проекта (банковский кейс)

Файлы `.csv` / `.zip` **не коммитятся** (см. `.gitignore`).

## Ожидаемая раскладка

```text
data/
├── raw/
│   └── train_ver2.csv          # исходный датасет курса
├── processed/                  # артефакты пайплайна (DVC)
└── README.md
```

## Как положить данные

Автоматически (рекомендуется):

```bash
bash scripts/download_dataset.sh
# перезаписать: FORCE=1 bash scripts/download_dataset.sh
```

Скрипт: [`scripts/download_dataset.sh`](../scripts/download_dataset.sh) — скачивает с  
https://disk.yandex.com/d/Io0siOESo2RAaA и кладёт `train_ver2.csv` в `data/raw/`.

Вручную:

1. Скачать архив с Яндекс.Диска курса: https://disk.yandex.com/d/Io0siOESo2RAaA
2. Распаковать `train_ver2.csv` в `data/raw/`

Если файл уже лежит в `data/bank/train_ver2.csv`, можно скопировать:

```bash
# Windows PowerShell
New-Item -ItemType Directory -Force data/raw | Out-Null
Copy-Item data/bank/train_ver2.csv data/raw/train_ver2.csv

# Linux / macOS
# mkdir -p data/raw && cp data/bank/train_ver2.csv data/raw/
```
