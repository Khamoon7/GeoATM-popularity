from __future__ import annotations

from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from app.db.models import RequestLog


class RequestLogRepository:
    """
    Репозиторий для работы с историей запросов.

    Инкапсулирует операции записи, чтения и очистки
    таблицы request_log.
    """

    def __init__(self, db: Session):
        """
        Инициализирует репозиторий с активной DB-сессией.
        """
        self.db = db

    def add_log(
        self,
        *,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        request_payload: str | None = None,
        response_payload: str | None = None,
        error: str | None = None,
        json_size_bytes: int | None = None,
        json_num_fields: int | None = None,
        address_len: int | None = None,
        address_tokens: int | None = None,
    ) -> RequestLog:
        """
        Сохраняет запись о запросе в базе данных.

        Возвращает созданный ORM-объект.
        """
        row = RequestLog(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=float(latency_ms),
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
        """
        Возвращает историю запросов с пагинацией.

        Записи сортируются по убыванию id.
        """
        stmt = (
            select(RequestLog)
            .order_by(RequestLog.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.execute(stmt).scalars().all())

    def clear_history(self) -> int:
        """
        Удаляет все записи истории запросов.

        Возвращает количество удалённых строк.
        """
        stmt = delete(RequestLog)
        res = self.db.execute(stmt)
        self.db.commit()
        return int(res.rowcount or 0)
