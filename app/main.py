from fastapi import FastAPI
from app.api.routes_forward import router as forward_router
from app.api.routes_history import router as history_router
from app.api.routes_stats import router as stats_router

# Создаём экземпляр приложения FastAPI
app = FastAPI(
    title="ATM Popularity Service",
    description="Сервис оценки популярности банкоматов",
    version="1.0.0",
)

# Подключаем роутеры к приложению
app.include_router(forward_router, tags=["forward"])
app.include_router(history_router, tags=["history"])
app.include_router(stats_router, tags=["stats"])

@app.get("/health")
def health():
    """
    Служебный эндпоинт.

    Зачем нужен:
    быстро проверить, что сервис запущен и отвечает

    Что возвращает:
    - JSON со статусом
    """
    return {"status": "ok"}


@app.get("/model_info")
def model_info():
    """
    Служебный эндпоинт.

    Зачем нужен:
    быстро проверить, какая модель (и какая версия) используется на бэкенде

    Сейчас возвращаем статический JSON.
    """
    return {
        "model_name": "ATM Popularity Model",
        "model_version": "1.0.0",
    }
