from typing import Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session
from app.db.models import RequestLog

def _compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """
    Вспомогательная функция: считает метрики для списка чисел.

    Что считаем:
    - mean: среднее значение
    - p50: медиана (50-й перцентиль)
    - p95: 95-й перцентиль
    - p99: 99-й перцентиль
    """

    # Если значений нет (например, таблица RequestLog пустая),то статистику считать нельзя — возвращаем None.
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    # Превращаем список в numpy-массив
    arr = np.array(values, dtype=float)

    # Считаем метрики и приводим к float, чтобы не было numpy-типов в ответе
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def get_stats(db: Session) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Основная функция сервиса статистики.

    На вход:
    - db: SQLAlchemy

    На выход:
    - словарь со статистикой, который потом вернётся пользователю через /stats

    Здесь мы:
    1) достаём все RequestLog
    2) берём из них latency_ms и json_num_fields
    3) считаем статистики
    """

    # Достаём все логи запросов из таблицы RequestLog
    logs = db.query(RequestLog).all()

    # Собираем список задержек (latency_ms), пропуски (None) не берём
    latency_values = [
        log.latency_ms for log in logs if log.latency_ms is not None
    ]

    # Собираем список "размеров" входного JSON
    json_fields_values = [
        log.json_num_fields for log in logs if log.json_num_fields is not None
    ]

    # Формируем итоговый ответ:
    # статистика по latency_ms
    # статистика по json_num_fields
    # сколько всего запросов в истории
    return {
        "latency_ms": _compute_stats(latency_values),
        "json_num_fields": _compute_stats(json_fields_values),
        "total_requests": len(logs),
    }