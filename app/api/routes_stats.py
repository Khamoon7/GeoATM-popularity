from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.stats_service import get_stats

router = APIRouter(tags=["stats"])


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    """
    Возвращает агрегированную статистику по истории запросов.

    Используется для мониторинга качества сервиса
    и анализа характеристик входных данных.
    """
    return get_stats(db)
