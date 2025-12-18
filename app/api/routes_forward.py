# app/api/routes_forward.py

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from app.core.schemas import ATMPredictRequest, ATMPredictResponse, Coords
from app.service.forward_service import ForwardService
from app.api.deps import get_forward_service

router = APIRouter()
history_logger = logging.getLogger("history")


def _count_json_fields(obj: Any) -> int:
    """Рекурсивно считает количество ключей во всех вложенных dict."""
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
async def forward(
    request: Request,
    svc: ForwardService = Depends(get_forward_service),
) -> ATMPredictResponse:
    start_ts = time.perf_counter()

    raw = await request.body()
    json_size_bytes = len(raw)

    #Парсинг инпута, при ошибке выдается код 400
    try:
        payload: Optional[Dict[str, Any]] = json.loads(raw.decode("utf-8")) if raw else None
    except Exception:
        raise HTTPException(status_code=400, detail="bad request")

    # Валидация инпута, при ошибке код 400
    try:
        req_obj = ATMPredictRequest.model_validate(payload)
    except (ValidationError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="bad request")

    json_num_fields = _count_json_fields(payload)
    # Логируем информацию о запросе
    history_logger.info(
        "forward_start",
        extra={
            "json_size_bytes": json_size_bytes,
            "json_num_fields": json_num_fields,
            "atm_id": req_obj.atm_id,
        },
    )

    # Пайплайн (геокодинг -> фичи -> валидация -> модель)
    try:
        result = await svc.run(req_obj.model_dump())
    except Exception:
        history_logger.exception("forward_pipeline_failed")
        # При проблеме с моделью выдается код 403
        raise HTTPException(status_code=403, detail="Модель не смогла обработать данные")

    # Если вернулся ok=False выдается код 400
    if not result.get("ok", False):
        history_logger.info(
            "forward_bad_request",
            extra={"atm_id": req_obj.atm_id, "error": result.get("error")},
        )
        raise HTTPException(status_code=400, detail="bad request")

    pred = float(result["popularity_index"])
    segment = _segment_from_prediction(pred)
    coords = Coords(lat=float(result["lat"]), lon=float(result["lon"]))

    resp = ATMPredictResponse(
        atm_id=req_obj.atm_id,
        popularity_index=pred,
        segment=segment,
        coords=coords,
    )

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    history_logger.info(
        "forward_finish",
        extra={"elapsed_ms": elapsed_ms, "segment": segment, "atm_id": resp.atm_id},
    )

    return resp