from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# URL подключения к базе данных
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./geoatm.db")

# Специальные аргументы для SQLite
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# SQLAlchemy engine приложения
engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args=connect_args,
)

# Фабрика сессий SQLAlchemy
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency FastAPI для получения DB-сессии.

    Открывает сессию на время запроса и гарантированно
    закрывает её после завершения обработки.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
