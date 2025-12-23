from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import RequestLog


Number = Union[int, float]


def _compute_stats(values: Sequence[Number]) -> Dict[str, Optional[float]]:
    """
    Вычисляет базовые агрегаты для числового ряда.

    Используется для расчёта статистик по логам.
    """
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None}

    arr = np.asarray(values, dtype=float)

    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def get_stats(db: Session) -> Dict[str, Any]:
    """
    Возвращает агрегированную статистику по истории запросов.

    Считает количество запросов и распределения
    ключевых числовых характеристик.
    """
    logs = db.query(RequestLog).all()

    # Сбор числовых рядов для агрегаций
    latency_values = [log.latency_ms for log in logs if log.latency_ms is not None]
    json_fields_values = [log.json_num_fields for log in logs if log.json_num_fields is not None]
    json_size_values = [log.json_size_bytes for log in logs if log.json_size_bytes is not None]
    address_len_values = [log.address_len for log in logs if log.address_len is not None]
    address_tokens_values = [log.address_tokens for log in logs if log.address_tokens is not None]

    return {
        "total_requests": len(logs),
        "latency_ms": _compute_stats(latency_values),
        "json_num_fields": _compute_stats(json_fields_values),
        "json_size_bytes": _compute_stats(json_size_values),
        "address_len": _compute_stats(address_len_values),
        "address_tokens": _compute_stats(address_tokens_values),
    }
