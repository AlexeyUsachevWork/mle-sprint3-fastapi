#!/usr/bin/env bash
# Скачивание train_ver2.csv с Яндекс.Диска курса в data/raw/.
#
# Использование:
#   bash scripts/download_dataset.sh
#   FORCE=1 bash scripts/download_dataset.sh          # перезаписать
#   DATASET_URL=https://disk.yandex.com/d/... bash scripts/download_dataset.sh
#
# Нужны: curl, python3 (для JSON / распаковки zip).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
TARGET="${RAW_DIR}/train_ver2.csv"
PUBLIC_URL="${DATASET_URL:-https://disk.yandex.com/d/Io0siOESo2RAaA}"
FORCE="${FORCE:-0}"
API="https://cloud-api.yandex.net/v1/disk/public/resources/download"

mkdir -p "${RAW_DIR}"

if [[ -f "${TARGET}" && "${FORCE}" != "1" ]]; then
  echo "Уже есть: ${TARGET}"
  echo "Чтобы скачать заново: FORCE=1 bash scripts/download_dataset.sh"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Нужен curl" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
  echo "Нужен python или python3" >&2
  exit 1
fi
PY="$(command -v python3 || command -v python)"

echo "Публичная ссылка: ${PUBLIC_URL}"
echo "Запрашиваю прямую ссылку у API Яндекс.Диска…"

ENCODED="$("${PY}" -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "${PUBLIC_URL}")"
HREF="$(curl -fsSL "${API}?public_key=${ENCODED}" | "${PY}" -c "import sys,json; print(json.load(sys.stdin)['href'])")"

TMP="${RAW_DIR}/.yadisk_download.tmp"
trap 'rm -f "${TMP}"' EXIT

echo "Скачиваю в ${RAW_DIR} (файл большой — подождите)…"
curl -fL --progress-bar -o "${TMP}" "${HREF}"

echo "Раскладываю train_ver2.csv…"
"${PY}" - "${TMP}" "${TARGET}" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)

if zipfile.is_zipfile(src):
    with zipfile.ZipFile(src) as zf:
        names = [n for n in zf.namelist() if n.rstrip("/").endswith("train_ver2.csv")]
        if not names:
            raise SystemExit(
                "В архиве нет train_ver2.csv. Содержимое:\n  " + "\n  ".join(zf.namelist()[:40])
            )
        with zf.open(names[0]) as fin, dst.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"Извлечено из zip: {names[0]} → {dst}")
else:
    # Иногда отдаётся сам CSV
    shutil.move(str(src), str(dst))
    print(f"Сохранено как CSV: {dst}")
PY

# Если python сделал move, TMP уже нет — trap ок
rm -f "${TMP}" 2>/dev/null || true

BYTES="$("${PY}" -c "from pathlib import Path; print(Path(r'''${TARGET}''').stat().st_size)")"
echo "Готово: ${TARGET} (${BYTES} bytes)"
