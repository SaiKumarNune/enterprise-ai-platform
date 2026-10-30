from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.core.config import settings

app = FastAPI(title=settings.app_title)
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(search_router)
