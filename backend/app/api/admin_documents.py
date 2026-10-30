from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.core.database import get_db
from app.models.user import ADMIN, User
from app.schemas.quarantine import AssignOwnerRequest, QuarantineDocumentResponse
from app.services.authorization import Principal
from app.services.quarantine import QuarantineError, assign_quarantined_owner, list_quarantined_documents

router = APIRouter()
TenantAdmin = Annotated[User, Depends(require_roles(ADMIN))]


def _http(exc: QuarantineError) -> HTTPException:
    codes = {
        "forbidden": status.HTTP_403_FORBIDDEN,
        "not_found": status.HTTP_404_NOT_FOUND,
        "conflict": status.HTTP_409_CONFLICT,
    }
    return HTTPException(status_code=codes.get(exc.code, status.HTTP_400_BAD_REQUEST), detail=exc.message)


@router.get("/admin/documents/quarantine", response_model=list[QuarantineDocumentResponse])
def list_quarantine(
    user: TenantAdmin,
    db: Session = Depends(get_db),
) -> list:
    try:
        return list_quarantined_documents(db, Principal.from_user(user))
    except QuarantineError as exc:
        raise _http(exc) from exc


@router.post(
    "/admin/documents/{document_id}/assign-owner",
    response_model=QuarantineDocumentResponse,
)
def assign_owner(
    document_id: UUID,
    payload: AssignOwnerRequest,
    user: TenantAdmin,
    db: Session = Depends(get_db),
) -> object:
    try:
        return assign_quarantined_owner(
            db,
            Principal.from_user(user),
            document_id=document_id,
            owner_id=payload.user_id,
        )
    except QuarantineError as exc:
        raise _http(exc) from exc
