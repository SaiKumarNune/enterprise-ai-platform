import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.admin_documents import router as admin_documents_router
from app.api.agents import router as agents_router
from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.rag import router as rag_router
from app.api.search import router as search_router
from app.core.config import settings
from app.core.logging import configure_logging, log_event
from app.core.observability import ObservabilityMiddleware
from app.core.request_context import REQUEST_ID_HEADER, get_request_id

configure_logging()
logger = logging.getLogger("app")

app = FastAPI(title=settings.app_title)
app.add_middleware(ObservabilityMiddleware)
app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(auth_router)
app.include_router(admin_documents_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(rag_router)
app.include_router(agents_router)


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    log_event(logger, "db.error", level=logging.ERROR, error_type=type(exc).__name__)
    response = JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database unavailable"},
    )
    request_id = get_request_id()
    if request_id:
        response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        raise exc
    log_event(logger, "http.unhandled_error", level=logging.ERROR, error_type=type(exc).__name__)
    response = JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
    request_id = get_request_id()
    if request_id:
        response.headers[REQUEST_ID_HEADER] = request_id
    return response
