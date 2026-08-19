from fastapi import FastAPI

'''
Шаг 3. Сервис Event Store
Чтобы выполнить второй пункт алгоритма 
(«для онлайн-взаимодействия пользователя с каким-то объектом 
можно использовать список похожих на него объектов»), 
необходим компонент, умеющий сохранять и 
выдавать последние события пользователя, — это Event Store. 

Реализуем его также в виде сервиса. 

В данном случае под взаимодействием пользователя с объектом будем 
подразумевать любое положительное событие, например: 
просмотр страницы с книгой, лайк, добавление в избранное и т. п.

Задание 3 из 6
Дополните код сервиса так, чтобы он по методу /put сохранял пару значений user_id и item_id как событие, а по методу /get возвращал события (первыми — самые последние).
'''


class EventStore:

    def __init__(self, max_events_per_user=10):

        self.events = {}
        self.max_events_per_user = max_events_per_user

    def put(self, user_id, item_id):
        """
        Сохраняет событие
        """

        # изменение: забираем уже накопленные события пользователя (если их нет — пустой список)
        user_events = self.events.get(user_id, [])
        self.events[user_id] = [item_id] + user_events[: self.max_events_per_user]

    def get(self, user_id, k):
        """
        Возвращает события для пользователя
        """
        # изменение: достаём список событий и возвращаем первые k (самые последние)
        user_events = self.events.get(user_id, [])
        return user_events[:k]

events_store = EventStore(max_events_per_user=10)

# создаём приложение FastAPI
app = FastAPI(title="events")

@app.post("/put")
async def put(user_id: int, item_id: int):
    """
    Сохраняет событие для user_id, item_id
    """

    events_store.put(user_id, item_id)

    return {"result": "ok"}

@app.post("/get")
async def get(user_id: int, k: int = 10):
    """
    Возвращает список последних k событий для пользователя user_id
    """

    events = events_store.get(user_id, k)

    return {"events": events}