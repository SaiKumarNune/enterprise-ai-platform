from typing import Annotated
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import AuthConfigError, create_access_token
from app.schemas.auth import TokenResponse
from app.services.oidc import (
    OidcDisabledError,
    OidcError,
    OidcProvider,
    begin_login,
    finish_login,
    get_oidc_provider as load_oidc_provider,
    oidc_is_configured,
)

router = APIRouter()


def get_oidc_provider() -> OidcProvider:
    try:
        return load_oidc_provider()
    except OidcDisabledError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.message) from exc
    except AuthConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OIDC is not configured",
        ) from exc


def _http(exc: OidcError) -> HTTPException:
    codes = {
        "disabled": status.HTTP_404_NOT_FOUND,
        "invalid_request": status.HTTP_400_BAD_REQUEST,
        "invalid_state": status.HTTP_401_UNAUTHORIZED,
        "invalid_nonce": status.HTTP_401_UNAUTHORIZED,
        "invalid_token": status.HTTP_401_UNAUTHORIZED,
        "expired": status.HTTP_401_UNAUTHORIZED,
        "missing_email": status.HTTP_401_UNAUTHORIZED,
        "unverified_email": status.HTTP_401_UNAUTHORIZED,
        "inactive": status.HTTP_401_UNAUTHORIZED,
        "missing_invite": status.HTTP_403_FORBIDDEN,
        "conflict": status.HTTP_409_CONFLICT,
        "discovery": status.HTTP_503_SERVICE_UNAVAILABLE,
        "token_exchange": status.HTTP_503_SERVICE_UNAVAILABLE,
        "jwks": status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    headers = {"WWW-Authenticate": "Bearer"} if codes.get(exc.code) == 401 else None
    return HTTPException(status_code=codes.get(exc.code, status.HTTP_400_BAD_REQUEST), detail=exc.message, headers=headers)


@router.get("/auth/oidc/login")
def oidc_login(
    db: Session = Depends(get_db),
    provider: OidcProvider = Depends(get_oidc_provider),
) -> RedirectResponse:
    if not settings.oidc_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OIDC login is not enabled")
    if not oidc_is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OIDC is not configured",
        )
    try:
        url = begin_login(db, provider)
    except OidcDisabledError as exc:
        raise _http(exc) from exc
    except AuthConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        ) from exc
    except OidcError as exc:
        raise _http(exc) from exc
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OIDC is not configured",
        )
    return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)


@router.get("/auth/oidc/callback", response_model=TokenResponse)
def oidc_callback(
    code: Annotated[str, Query(min_length=1, max_length=4096)],
    state: Annotated[str, Query(min_length=1, max_length=256)],
    db: Session = Depends(get_db),
    provider: OidcProvider = Depends(get_oidc_provider),
) -> TokenResponse:
    if not settings.oidc_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OIDC login is not enabled")
    try:
        user = finish_login(db, provider, code=code, state=state)
        token = create_access_token(user.id)
    except OidcDisabledError as exc:
        raise _http(exc) from exc
    except AuthConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        ) from exc
    except OidcError as exc:
        raise _http(exc) from exc
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.jwt_expire_minutes * 60,
    )
