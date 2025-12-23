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

router = APIRouter(tags=["forward"])


def _count_json_fields(obj: Any) -> int:
    if isinstance(obj, dict):
        return len(obj) + sum(_count_json_fields(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_json_fields(x) for x in obj)
    return 0


def _json_size_bytes(payload: Dict[str, Any]) -> int:
    try:
        raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        return len(raw)
    except Exception:
        return 0


def _address_len(payload: Dict[str, Any]) -> Optional[int]:
    addr = payload.get("address")
    if isinstance(addr, str):
        return len(addr)
    return None


def _address_tokens(payload: Dict[str, Any]) -> Optional[int]:
    addr = payload.get("address")
    if not isinstance(addr, str):
        return None
    parts = [t for t in addr.strip().split() if t]
    return len(parts)


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

    endpoint = "/forward"
    method = "POST"

    status_code: int = 200
    response_obj: Optional[Dict[str, Any]] = None
    error_text: Optional[str] = None

    payload: Dict[str, Any] = req_obj.model_dump()

    json_size_bytes = _json_size_bytes(payload)
    json_num_fields = _count_json_fields(payload)
    address_len = _address_len(payload)
    address_tokens = _address_tokens(payload)

    try:
        result = await svc.run(payload)

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

    except HTTPException as e:
        status_code = int(e.status_code)
        error_text = str(e.detail)
        raise

    except ValueError as e:
        status_code = 400
        error_text = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=400, detail="bad request")

    except Exception as e:
        status_code = 403
        error_text = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")

    finally:
        latency_ms = (time.perf_counter() - start_ts) * 1000.0

        log_request(
            db,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            request_payload=payload,
            response_payload=response_obj,
            error=error_text,
            json_size_bytes=json_size_bytes,
            json_num_fields=json_num_fields,
            address_len=address_len,
            address_tokens=address_tokens,
        )