import logging

from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI

from recommendations import Recommendations

logger = logging.getLogger("uvicorn.error")

rec_store = Recommendations()

# URL сервисов Event Store и Feature Store
events_store_url = "http://127.0.0.1:8020"
features_store_url = "http://127.0.0.1:8010"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # код ниже (до yield) выполнится только один раз при запуске сервиса
    logger.info("Starting")

    rec_store.load(
        "personal",
        "final_recommendations_feat.parquet",
        columns=["user_id", "item_id", "rank"],
    )
    rec_store.load(
        "default",
        "top_recs.parquet",
        columns=["item_id", "rank"],
    )

    yield
    # этот код выполнится только один раз при остановке сервиса
    logger.info("Stopping")


# создаём приложение FastAPI
app = FastAPI(title="recommendations", lifespan=lifespan)


def dedup_ids(ids):
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение
    """
    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]

    return ids


@app.post("/recommendations_offline")
async def recommendations_offline(user_id: int, k: int = 100):
    """
    Возвращает список офлайн-рекомендаций длиной k для пользователя user_id
    """

    recs = rec_store.get(user_id=user_id, k=k)

    return {"recs": recs}


@app.post("/recommendations_online")
async def recommendations_online(user_id: int, k: int = 100):
    """
    Возвращает список онлайн-рекомендаций длиной k для пользователя user_id
    """

    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    # получаем список последних событий пользователя, возьмём три последних
    params = {"user_id": user_id, "k": 3}
    # изменение: запрашиваем последние 3 события из Event Store
    resp = requests.post(events_store_url + "/get", headers=headers, params=params)
    events = resp.json()
    events = events.get("events", [])

    # получаем список айтемов, похожих на последние три, с которыми взаимодействовал пользователь
    items = []
    scores = []
    for item_id in events:
        # для каждого item_id получаем список похожих в item_similar_items
        # изменение: запрашиваем похожие объекты у Feature Store
        params = {"item_id": item_id, "k": k}
        resp = requests.post(
            features_store_url + "/similar_items",
            headers=headers,
            params=params,
        )
        item_similar_items = resp.json()
        items += item_similar_items["item_id_2"]
        scores += item_similar_items["score"]
    # сортируем похожие объекты по scores в убывающем порядке
    # для старта это приемлемый подход
    combined = list(zip(items, scores))
    combined = sorted(combined, key=lambda x: x[1], reverse=True)
    combined = [item for item, _ in combined]

    # удаляем дубликаты, чтобы не выдавать одинаковые рекомендации
    recs = dedup_ids(combined)[:k]

    return {"recs": recs}


@app.post("/recommendations")
async def recommendations(user_id: int, k: int = 100):
    """
    Возвращает список рекомендаций длиной k для пользователя user_id
    """

    # изменение: получаем офлайн- и онлайн-рекомендации
    recs_offline = await recommendations_offline(user_id, k)
    recs_online = await recommendations_online(user_id, k)

    recs_offline = recs_offline["recs"]
    recs_online = recs_online["recs"]

    recs_blended = []

    min_length = min(len(recs_offline), len(recs_online))
    # чередуем элементы из списков, пока позволяет минимальная длина
    for i in range(min_length):
        # изменение: сначала онлайн, затем офлайн
        recs_blended.append(recs_online[i])
        recs_blended.append(recs_offline[i])

    # добавляем оставшиеся элементы в конец
    # изменение: хвосты обоих списков после min_length
    recs_blended += recs_online[min_length:]
    recs_blended += recs_offline[min_length:]

    # удаляем дубликаты
    recs_blended = dedup_ids(recs_blended)

    # оставляем только первые k рекомендаций
    # изменение: обрезаем итоговый список до k
    recs_blended = recs_blended[:k]

    return {"recs": recs_blended}
