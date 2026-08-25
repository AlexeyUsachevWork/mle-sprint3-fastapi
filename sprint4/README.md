# Подготовка ВМ для работы со спринтом

## Склонируйте репозиторий

Склонируйте на свою ВМ начальный репоизиторий:

```
git clone git@github.com:practicum-mle/mle-recsys-start.git
```

## Поставьте Python-пакеты

Для работы со спринтом используются Python-пакеты, которые могли не встретиться в предыдущих спринтах, и поэтому вероятно отсутствующие в системе. Данные пакеты следует поставить. Для чего рекомендуется создать новое окружения, и, используя файл `requirements.txt` из директории репозитория, установить их.

Создать новое виртуальное окружение (за пределами директории репозитория) можно командой:

```
python3 -m venv env_recsys_start
```

Это виртуальное окружение будем называть виртуальным окружением спринта.

После его инициализации следующей командой

```
. env_recsys_start/bin/activate
```

установите в него необходимые Python-пакеты следующей командой

```
pip install -r requirements.txt
```

### Скачайте файлы с данными

Для начала работы понадобится два файла с данными:
- [books.parquet](https://storage.yandexcloud.net/mle-data/goodsread/books.parquet)
- [interactions.parquet](https://storage.yandexcloud.net/mle-data/goodsread/interactions.parquet)

Позже, вам понадобятся ещё файлы:

- [candidates/inference/als_recommendations.parquet](https://storage.yandexcloud.net/mle-data/candidates/inference/als_recommendations.parquet)
- [candidates/inference/content_recommendations.parquet](https://storage.yandexcloud.net/mle-data/candidates/inference/content_recommendations.parquet)
- [candidates/training/als_recommendations.parquet](https://storage.yandexcloud.net/mle-data/candidates/training/als_recommendations.parquet)
- [candidates/training/content_recommendations.parquet](https://storage.yandexcloud.net/mle-data/candidates/training/content_recommendations.parquet)
- [top_recs.parquet](https://storage.yandexcloud.net/mle-data/top_recs.parquet)
 
Скачайте их в директорию локального репозитория. Для удобства вы можете воспользоваться командой wget:

```
wget https://storage.yandexcloud.net/mle-data/goodsread/books.parquet

wget https://storage.yandexcloud.net/mle-data/goodsread/interactions.parquet
```

и

```
mkdir -p candidates/inference
wget https://storage.yandexcloud.net/mle-data/candidates/inference/als_recommendations.parquet -P candidates/inference
wget https://storage.yandexcloud.net/mle-data/candidates/inference/content_recommendations.parquet -P candidates/inference

mkdir -p candidates/training
wget https://storage.yandexcloud.net/mle-data/candidates/training/als_recommendations.parquet -P candidates/training
wget https://storage.yandexcloud.net/mle-data/candidates/training/content_recommendations.parquet -P candidates/training

wget https://storage.yandexcloud.net/mle-data/top_recs.parquet
```


## Откройте шаблон ноутбука

Запустите Jupyter Lab

```
jupyter lab --ip=0.0.0.0 --no-browser
```

Откройте в нём файл `preprocessing.ipynb`. Это ноутбук, в котором предлагается провести предварительную подготовку данных.

После выполнения подготовки данных, их сохранения, дальнейшую работу предлагается вести в ноутбуке `main_offline.ipynb`.

