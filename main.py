from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse

from dotenv import load_dotenv

from app.db.session import engine
from app.db.models import Base

from app.api.routes_forward import router as forward_router
from app.api.routes_history import router as history_router
from app.api.routes_stats import router as stats_router

load_dotenv()

app = FastAPI(
    title="ATM Popularity Service",
    description="Сервис оценки популярности банкоматов",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


app.include_router(forward_router)
app.include_router(history_router)
app.include_router(stats_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model_info")
def model_info():
    return {
        "model_name": "ATM Popularity Model",
        "model_version": "1.0.0",
    }


def _is_forward(request: Request) -> bool:
    return request.url.path.rstrip("/") == "/forward"


# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError) -> PlainTextResponse:
#     if _is_forward(request):
#         return PlainTextResponse("bad request", status_code=400)

#     return PlainTextResponse("bad request", status_code=400)


# @app.exception_handler(HTTPException)
# async def http_exception_handler(request: Request, exc: HTTPException) -> PlainTextResponse:
#     if _is_forward(request):
#         if exc.status_code == 400:
#             return PlainTextResponse("bad request", status_code=400)
#         if exc.status_code == 403:
#             return PlainTextResponse("модель не смогла обработать данные", status_code=403)

#     return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


# @app.exception_handler(Exception)
# async def unhandled_exception_handler(request: Request, exc: Exception) -> PlainTextResponse:
#     if _is_forward(request):
#         return PlainTextResponse("модель не смогла обработать данные", status_code=403)

#     return PlainTextResponse("internal server error", status_code=500)
