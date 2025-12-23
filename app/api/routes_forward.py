from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_forward_service
from app.core.schemas import ATMPredictRequest, ATMPredictResponse, Coords
from app.db.session import get_db
from app.services.forward_service import ForwardService
from app.services.history_logger import log_request

router = APIRouter()


def _count_json_fields(obj: Any) -> int:
    if isinstance(obj, dict):
        return len(obj) + sum(_count_json_fields(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_json_fields(x) for x in obj)
    return 0


def _segment_from_prediction(pred: float) -> str:
    if pred > 0:
        return "high"
    if pred < 0:
        return "low"
    return "med"


@router.post("/forward", response_model=ATMPredictResponse)
async def forward(
    req_obj: ATMPredictRequest,
    svc: ForwardService = Depends(get_forward_service),
    db: Session = Depends(get_db),
) -> ATMPredictResponse:
    start_ts = time.perf_counter()

    status_code = 200
    response_obj: Optional[Dict[str, Any]] = None
    error_text: Optional[str] = None

    endpoint = "/forward"
    method = "POST"

    payload: Dict[str, Any] = req_obj.model_dump()

    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    _json_size_bytes = len(raw)
    _json_num_fields = _count_json_fields(payload)

    try:
        try:
            result = await svc.run(payload)
        except Exception as e:
            status_code = 500
            error_text = f"{type(e).__name__}: {e}"
            raise HTTPException(
                status_code=500,
                detail="Модель не смогла обработать данные",
            )

        if not result.get("ok", False):
            status_code = 400
            error_text = f"bad request (pipeline ok=false): {result.get('error')}"
            raise HTTPException(status_code=400, detail="bad request")

        pred = float(result["popularity_index"])
        segment = _segment_from_prediction(pred)

        resp = ATMPredictResponse(
            atm_id=req_obj.atm_id,
            popularity_index=pred,
            segment=segment,
            coords=Coords(
                lat=float(result["lat"]),
                lon=float(result["lon"]),
            ),
        )

        response_obj = resp.model_dump()
        return resp

    finally:
        latency_ms = int((time.perf_counter() - start_ts) * 1000)

        log_request(
            db,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            request_payload=payload,
            response_payload=response_obj,
            error=error_text,
        )
