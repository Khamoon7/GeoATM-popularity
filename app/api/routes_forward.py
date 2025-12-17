# app/api/routes_forward.py

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.core.schemas import ATMPredictRequest, ATMPredictResponse
from app.core.model import atm_model_service

router = APIRouter()
history_logger = logging.getLogger("history") 


def _count_json_fields(obj: Any) -> int:
    """Рекурсивно считает количество ключей во всех вложенных dict ."""
    if isinstance(obj, dict):
        return len(obj) + sum(_count_json_fields(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_json_fields(x) for x in obj)
    return 0


def _segment_from_prediction(pred: float) -> str:
    # Предсказание >0 => high, <0 => low
    if pred > 0:
        return "high"
    if pred < 0:
        return "low"
    return "med"


@router.post("/forward", response_model=ATMPredictResponse)
async def forward(request: Request) -> ATMPredictResponse:
    start_ts = time.perf_counter()

    raw = await request.body()
    json_size_bytes = len(raw)

    # Парсинг JSON, любой невалидный => 400 "bad request"
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else None
    except Exception:
        raise HTTPException(status_code=400, detail="bad request")

    # Валидация схемы: любые ошибки => 400 "bad request"
    try:
        req_obj = ATMPredictRequest.model_validate(payload)
    except (ValidationError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="bad request")

    json_num_fields = _count_json_fields(payload)

    # логгер истории
    history_logger.info(
        "forward_start",
        extra={
            "json_size_bytes": json_size_bytes,
            "json_num_fields": json_num_fields,
            "atm_id": getattr(req_obj, "atm_id", None),
        },
    )

    # Инференс, любое исключение => 403 "модель не смогла обработать данные"
    try:
        from app.core.model import model  # type: ignore

        # model.predict_popularity должен вернуть индекс популярности (float)
        pred = float(atm_model_service.predict_popularity(req_obj.model_dump()))
    except Exception:
        history_logger.exception("forward_inference_failed")
        raise HTTPException(status_code=403, detail="Модель не смогла обработать данные")

    segment = _segment_from_prediction(pred)

    resp = ATMPredictResponse(
        popularity_index=pred,
        segment=segment,  # low/med/high
        coords=req_obj.coords,
    )

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    history_logger.info(
        "forward_finish",
        extra={"elapsed_ms": elapsed_ms, "segment": segment},
    )

    return resp