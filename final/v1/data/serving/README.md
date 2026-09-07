# Serving features store (lookup по ncodpers для API).
#
# Собрать:
#   python -m scripts.export_serving_features
#
# Файлы:
#   clients_features.parquet — один ряд на клиента
#   meta.json — источник и число клиентов
#
# В Docker каталог монтируется как /app/data/serving (см. docker-compose.yml).
