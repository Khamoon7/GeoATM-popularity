from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.stats_service import get_stats


# Создаём роутер
# Он будет подключён в main.py
router = APIRouter()

@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    """
    Эндпоинт GET /stats.

    Назначение:
    Возвращает агрегированную статистику по истории запросов сервиса.

    Как работает:
    1) FastAPI автоматически вызывает get_db() и передаёт сюда db-сессию.
    2) Мы передаём эту сессию в stats_service.get_stats().
    3) Получаем словарь со статистикой.
    4) Возвращаем его клиенту (FastAPI сам сериализует в JSON).
    """

    # Вызываем сервисный слой для подсчёта статистики
    stats_data = get_stats(db)

    # Возвращаем результат
    return stats_data
