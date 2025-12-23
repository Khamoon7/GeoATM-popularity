from __future__ import annotations

from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from app.db.models import RequestLog


class RequestLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_log(
            self,
            *,
            endpoint: str,
            method: str,
            status_code: int,
            latency_ms: int,
            request_payload: str | None = None,
            response_payload: str | None = None,
            error: str | None = None,
            json_size_bytes: int | None = None,
            json_num_fields: int | None = None,
            address_len: int | None = None,
            address_tokens: int | None = None,
    ) -> RequestLog:
        row = RequestLog(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            request_payload=request_payload,
            response_payload=response_payload,
            error=error,
            json_size_bytes=json_size_bytes,
            json_num_fields=json_num_fields,
            address_len=address_len,
            address_tokens=address_tokens,
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def get_history(self, *, limit: int = 50, offset: int = 0) -> list[RequestLog]:
        stmt = (
            select(RequestLog)
            .order_by(RequestLog.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.execute(stmt).scalars().all())

    def clear_history(self) -> int:
        stmt = delete(RequestLog)
        res = self.db.execute(stmt)
        self.db.commit()
        return int(res.rowcount or 0)
