from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.stats_service import get_stats

router = APIRouter()

@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    stats_data = get_stats(db)

    return stats_data
