from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by local development and future load balancers."""
    return {"status": "healthy", "service": settings.service_name}
