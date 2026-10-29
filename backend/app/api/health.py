from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by local development and future load balancers."""
    return {"status": "healthy", "service": settings.service_name}


@router.get("/db-health")
def db_health(db: Session = Depends(get_db)) -> dict[str, str]:
    """Readiness check that the application can reach PostgreSQL."""
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from None
    return {"status": "healthy", "database": "postgresql"}
