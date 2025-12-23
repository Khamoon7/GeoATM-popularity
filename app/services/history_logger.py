from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.db.repositories import RequestLogRepository


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def log_request(
    db: Session,
    *,
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: int,
    request_payload: Any | None = None,
    response_payload: Any | None = None,
    error: str | None = None,
    json_size_bytes: int | None = None,
    json_num_fields: int | None = None,
    address_len: int | None = None,
    address_tokens: int | None = None,
) -> None:
    repo = RequestLogRepository(db)

    repo.add_log(
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        latency_ms=latency_ms,
        request_payload=_safe_json_dumps(request_payload) if request_payload is not None else None,
        response_payload=_safe_json_dumps(response_payload) if response_payload is not None else None,
        error=error,
        json_size_bytes=json_size_bytes,
        json_num_fields=json_num_fields,
        address_len=address_len,
        address_tokens=address_tokens,
    )
