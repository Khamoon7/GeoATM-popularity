from fastapi import FastAPI

from app.db.session import engine
from app.db.models import Base
from app.api.routes_history import router as history_router

from app.api.routes_forward import router as forward_router
from app.api.routes_stats import router as stats_router

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="ATM Popularity Service",
    description="Сервис оценки популярности банкоматов",
    version="1.0.0",
)

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


app.include_router(forward_router, tags=["forward"])

app.include_router(stats_router, tags=["stats"])

app.include_router(history_router, tags=["history"])
@app.get("/health")
def health():

    return {"status": "ok"}


@app.get("/model_info")
def model_info():
    return {
        "model_name": "ATM Popularity Model",
        "model_version": "1.0.0",
    }
