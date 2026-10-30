from collections.abc import Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import AuthConfigError, decode_access_token
from app.models.user import User
from app.services.users import get_user_by_id

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)
UNCONFIGURED = HTTPException(
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    detail="Authentication is not configured",
)


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Session = Depends(get_db),
) -> User:
    try:
        user_id: UUID = decode_access_token(token)
    except AuthConfigError as exc:
        raise UNCONFIGURED from exc
    except ValueError as exc:
        raise UNAUTHORIZED from exc
    user = get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise UNAUTHORIZED
    return user


def require_roles(*roles: str) -> Callable[..., User]:
    allowed = frozenset(roles)

    def dependency(user: Annotated[User, Depends(get_current_user)]) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user

    return dependency
