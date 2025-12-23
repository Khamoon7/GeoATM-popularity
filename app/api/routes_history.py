from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db.repositories import RequestLogRepository
from app.db.session import get_db

router = APIRouter(tags=["history"])


@router.get("/history")
def get_history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    repo = RequestLogRepository(db)
    rows = repo.get_history(limit=limit, offset=offset)

    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "endpoint": r.endpoint,
            "method": r.method,
            "status_code": r.status_code,
            "latency_ms": r.latency_ms,
            "request_payload": r.request_payload,
            "response_payload": r.response_payload,
            "error": r.error,
            "json_size_bytes": r.json_size_bytes,
            "json_num_fields": r.json_num_fields,
            "address_len": r.address_len,
            "address_tokens": r.address_tokens,
        }
        for r in rows
    ]


@router.delete("/history")
def delete_history(
    db: Session = Depends(get_db),
    x_confirm_token: str | None = Header(default=None, alias="X-Confirm-Token"),
):
    expected = os.getenv("HISTORY_DELETE_TOKEN")
    if not expected:
        raise HTTPException(status_code=500, detail="HISTORY_DELETE_TOKEN is not set on server")

    if not x_confirm_token or x_confirm_token != expected:
        raise HTTPException(status_code=403, detail="Invalid confirm token")

    repo = RequestLogRepository(db)
    deleted = repo.clear_history()
    return {"deleted": deleted}
