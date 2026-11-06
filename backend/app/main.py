import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.sessions import SessionMiddleware

from app.api.admin_documents import router as admin_documents_router
from app.api.agents import router as agents_router
from app.api.auth import router as auth_router
from app.api.invitations import router as invitations_router
from app.api.oidc import router as oidc_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.rag import router as rag_router
from app.api.search import router as search_router
from app.core.config import settings
from app.core.logging import configure_logging, log_event
from app.core.observability import ObservabilityMiddleware
from app.core.request_context import REQUEST_ID_HEADER, get_request_id
from app.web.router import router as ui_router

configure_logging()
logger = logging.getLogger("app")

app = FastAPI(title=settings.app_title)
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.jwt_secret or "local-ui-session-unconfigured",
    same_site="lax",
    https_only=False,
    max_age=max(settings.jwt_expire_minutes, 1) * 60,
)
app.include_router(ui_router)
app.mount(
    "/app/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "web" / "static")),
    name="ui-static",
)
app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(auth_router)
app.include_router(oidc_router)
app.include_router(invitations_router)
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
