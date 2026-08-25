import requests

recommendations_url = "http://127.0.0.1:8000"
events_store_url = "http://127.0.0.1:8020"

headers = {"Content-type": "application/json", "Accept": "text/plain"}

user_id = 1291250
event_item_ids = [7144, 16299, 5907, 18135]

# 1) Генерируем онлайн-события
for event_item_id in event_item_ids:
    resp = requests.post(
        events_store_url + "/put",
        headers=headers,
        params={"user_id": user_id, "item_id": event_item_id},
    )
    print("put", event_item_id, resp.status_code, resp.json())

# 2) Получаем 10 рекомендаций каждого типа
params = {"user_id": user_id, "k": 10}

resp_offline = requests.post(
    recommendations_url + "/recommendations_offline",
    headers=headers,
    params=params,
)
resp_online = requests.post(
    recommendations_url + "/recommendations_online",
    headers=headers,
    params=params,
)
resp_blended = requests.post(
    recommendations_url + "/recommendations",
    headers=headers,
    params=params,
)

recs_offline = resp_offline.json()["recs"]
recs_online = resp_online.json()["recs"]
recs_blended = resp_blended.json()["recs"]

print("offline:", recs_offline)
print("online:", recs_online)
print("blended:", recs_blended)
