from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    Базовый класс для всех ORM-моделей проекта.
    """
    pass


class RequestLog(Base):
    """
    ORM-модель записи истории запросов.

    Хранит метаданные HTTP-запроса, характеристики входного JSON
    и результат обработки запроса сервисом.
    """
    __tablename__ = "request_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    method: Mapped[str] = mapped_column(String(16), nullable=False)

    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # Сырые JSON-пейлоады запроса и ответа
    request_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Агрегированные характеристики входного запроса
    json_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    json_num_fields: Mapped[int | None] = mapped_column(Integer, nullable=True)
    address_len: Mapped[int | None] = mapped_column(Integer, nullable=True)
    address_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
