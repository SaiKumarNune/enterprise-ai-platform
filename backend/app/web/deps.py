from __future__ import annotations

from pathlib import Path

from fastapi import Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import AuthConfigError, decode_access_token
from app.models.user import APPROVER_ROLES, User
from app.services.users import get_user_by_id

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

LOGIN_REDIRECT = HTTPException(
    status_code=status.HTTP_303_SEE_OTHER,
    headers={"Location": "/app/login"},
)


def is_hx(request: Request) -> bool:
    return request.headers.get("hx-request", "").lower() == "true"


def get_ui_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.session.get("access_token")
    if not token:
        raise LOGIN_REDIRECT
    try:
        user_id = decode_access_token(token)
    except AuthConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        ) from exc
    except ValueError as exc:
        request.session.clear()
        raise LOGIN_REDIRECT from exc
    user = get_user_by_id(db, user_id)
    if user is None or not user.is_active or user.tenant_id is None:
        request.session.clear()
        raise LOGIN_REDIRECT
    return user


def page_context(request: Request, user: User | None = None, **extra):
    return {
        "request": request,
        "user": user,
        "can_approve": bool(user is not None and user.role in APPROVER_ROLES),
        **extra,
    }


def error_detail(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, str) and detail.strip():
        return detail
    return "The request could not be completed."
