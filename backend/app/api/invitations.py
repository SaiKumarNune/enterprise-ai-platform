from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.core.database import get_db
from app.core.security import AuthConfigError
from app.models.invitation import Invitation
from app.models.user import ADMIN, User
from app.schemas.invitation import CreateInvitationRequest, CreatedInvitationResponse, InvitationResponse
from app.services.invitations import InvitationError, create_invitation, list_invitations

router = APIRouter()
TenantAdmin = Annotated[User, Depends(require_roles(ADMIN))]


@router.post(
    "/admin/invitations",
    response_model=CreatedInvitationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_invite(
    payload: CreateInvitationRequest,
    user: TenantAdmin,
    db: Session = Depends(get_db),
) -> CreatedInvitationResponse:
    try:
        invitation, token = create_invitation(
            db,
            actor=user,
            email=payload.email,
            ttl_hours=payload.expires_in_hours,
        )
    except AuthConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        ) from exc
    except InvitationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message) from exc
    return CreatedInvitationResponse(
        id=invitation.id,
        email=invitation.email,
        tenant_id=invitation.tenant_id,
        token=token,
        expires_at=invitation.expires_at,
        created_at=invitation.created_at,
    )


@router.get("/admin/invitations", response_model=list[InvitationResponse])
def list_invites(
    user: TenantAdmin,
    db: Session = Depends(get_db),
) -> list[Invitation]:
    return list_invitations(db, tenant_id=user.tenant_id)
