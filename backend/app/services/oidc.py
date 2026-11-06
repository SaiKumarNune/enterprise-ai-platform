from __future__ import annotations

import hashlib
import secrets
from base64 import urlsafe_b64encode
from datetime import datetime, timedelta, timezone
from typing import Protocol
from urllib.parse import urlencode
from uuid import uuid4

import httpx
import jwt
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import AuthConfigError, hash_secret, is_valid_email, normalize_email, require_jwt_config
from app.models.identity import ExternalIdentity
from app.models.oidc_state import OidcLoginState
from app.models.user import EMPLOYEE, User
from app.services.invitations import InvitationError, claim_open_invitation
from app.services.users import DuplicateUserError, create_employee, get_user_by_email, get_user_by_id

OIDC_ALGORITHMS = ("RS256",)
_STATE_TTL = timedelta(minutes=10)
_HTTP_TIMEOUT = 10.0


class OidcError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class OidcDisabledError(OidcError):
    def __init__(self) -> None:
        super().__init__("disabled", "OIDC login is not enabled")


class OidcProvider(Protocol):
    issuer: str
    client_id: str
    redirect_uri: str
    authorization_endpoint: str

    def authorization_url(self, *, state: str, nonce: str, code_challenge: str) -> str: ...

    def fetch_id_token(self, *, code: str, code_verifier: str) -> str: ...

    def signing_key(self, id_token: str): ...


def oidc_is_configured() -> bool:
    return bool(
        settings.oidc_enabled
        and settings.oidc_issuer.strip()
        and settings.oidc_client_id.strip()
        and settings.oidc_redirect_uri.strip()
    )


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def begin_login(db: Session, provider: OidcProvider) -> str:
    if not oidc_is_configured():
        raise OidcDisabledError()
    require_jwt_config()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier, challenge = _pkce_pair()
    now = datetime.now(timezone.utc)
    db.add(
        OidcLoginState(
            state_hash=hash_secret(state),
            nonce=nonce,
            code_verifier=verifier,
            expires_at=now + _STATE_TTL,
            created_at=now,
        )
    )
    db.commit()
    return provider.authorization_url(state=state, nonce=nonce, code_challenge=challenge)


def finish_login(db: Session, provider: OidcProvider, *, code: str, state: str) -> User:
    if not oidc_is_configured():
        raise OidcDisabledError()
    require_jwt_config()
    if not code.strip() or not state.strip():
        raise OidcError("invalid_request", "Missing authorization code or state")
    login_state = db.get(OidcLoginState, hash_secret(state))
    if login_state is None:
        raise OidcError("invalid_state", "Invalid or expired login state")
    db.delete(login_state)
    db.commit()
    now = datetime.now(timezone.utc)
    expires_at = login_state.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= now:
        raise OidcError("invalid_state", "Invalid or expired login state")
    id_token = provider.fetch_id_token(code=code.strip(), code_verifier=login_state.code_verifier)
    claims = validate_id_token(
        id_token,
        issuer=provider.issuer,
        audience=provider.client_id,
        nonce=login_state.nonce,
        key=provider.signing_key(id_token),
    )
    return map_verified_identity(db, claims)


def validate_id_token(
    token: str,
    *,
    issuer: str,
    audience: str,
    nonce: str,
    key,
    leeway: int = 30,
) -> dict:
    try:
        claims = jwt.decode(
            token,
            key=key,
            algorithms=list(OIDC_ALGORITHMS),
            audience=audience,
            issuer=issuer,
            leeway=leeway,
            options={
                "require": ["exp", "iat", "iss", "aud", "sub"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": True,
            },
        )
    except jwt.ExpiredSignatureError as exc:
        raise OidcError("expired", "Identity token has expired") from exc
    except jwt.PyJWTError as exc:
        raise OidcError("invalid_token", "Identity token is invalid") from exc
    if not secrets.compare_digest(str(claims.get("nonce") or ""), nonce):
        raise OidcError("invalid_nonce", "Identity token is invalid")
    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise OidcError("invalid_token", "Identity token is invalid")
    email = normalize_email(str(claims.get("email") or ""))
    if not is_valid_email(email):
        raise OidcError("missing_email", "Identity token is missing a verified email")
    if claims.get("email_verified") is False:
        raise OidcError("unverified_email", "Identity token email is not verified")
    claims["email"] = email
    claims["sub"] = subject
    return claims


def map_verified_identity(db: Session, claims: dict) -> User:
    issuer = str(claims["iss"])
    subject = str(claims["sub"])
    email = str(claims["email"])
    linked = db.scalar(
        select(ExternalIdentity).where(
            ExternalIdentity.issuer == issuer,
            ExternalIdentity.subject == subject,
        )
    )
    if linked is not None:
        user = get_user_by_id(db, linked.user_id)
        if user is None or not user.is_active:
            raise OidcError("inactive", "Account is not active")
        linked.email = email
        db.commit()
        db.refresh(user)
        return user

    if get_user_by_email(db, email) is not None:
        raise OidcError("conflict", "Email already registered")

    invitation = None
    if settings.auth_invite_only:
        try:
            invitation = claim_open_invitation(db, email=email, commit=False)
        except InvitationError as exc:
            raise OidcError("missing_invite", "Registration requires an invitation") from exc
    try:
        user = create_employee(
            db,
            email=email,
            password=None,
            tenant_id=invitation.tenant_id if invitation is not None else None,
            commit=False,
        )
    except DuplicateUserError as exc:
        raise OidcError("conflict", "Email already registered") from exc
    user.role = EMPLOYEE
    if invitation is not None:
        invitation.consumed_by = user.id
    _attach_identity(db, user=user, issuer=issuer, subject=subject, email=email, commit=False)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise OidcError("conflict", "Email already registered") from exc
    db.refresh(user)
    return user


def _attach_identity(
    db: Session,
    *,
    user: User,
    issuer: str,
    subject: str,
    email: str,
    commit: bool = True,
) -> None:
    db.add(
        ExternalIdentity(
            id=uuid4(),
            user_id=user.id,
            issuer=issuer,
            subject=subject,
            email=email,
        )
    )
    if commit:
        db.commit()
        db.refresh(user)


class HttpOidcProvider:
    def __init__(self) -> None:
        if not oidc_is_configured():
            raise OidcDisabledError()
        self.issuer = settings.oidc_issuer.strip().rstrip("/")
        self.client_id = settings.oidc_client_id.strip()
        self.client_secret = settings.oidc_client_secret
        self.redirect_uri = settings.oidc_redirect_uri.strip()
        self.scope = settings.oidc_scopes.strip() or "openid email"
        self._discovery: dict | None = None
        self._jwks_client: jwt.PyJWKClient | None = None

    @property
    def authorization_endpoint(self) -> str:
        return str(self._discover()["authorization_endpoint"])

    def authorization_url(self, *, state: str, nonce: str, code_challenge: str) -> str:
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "scope": self.scope,
                "state": state,
                "nonce": nonce,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }
        )
        return f"{self.authorization_endpoint}?{query}"

    def fetch_id_token(self, *, code: str, code_verifier: str) -> str:
        token_endpoint = str(self._discover()["token_endpoint"])
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        try:
            response = httpx.post(token_endpoint, data=data, timeout=_HTTP_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OidcError("token_exchange", "Could not exchange authorization code") from exc
        id_token = payload.get("id_token")
        if not id_token or not isinstance(id_token, str):
            raise OidcError("token_exchange", "Could not exchange authorization code")
        return id_token

    def signing_key(self, id_token: str):
        jwks_uri = str(self._discover()["jwks_uri"])
        if self._jwks_client is None or getattr(self._jwks_client, "uri", None) != jwks_uri:
            self._jwks_client = jwt.PyJWKClient(jwks_uri, lifespan=300, timeout=_HTTP_TIMEOUT)
        try:
            return self._jwks_client.get_signing_key_from_jwt(id_token).key
        except Exception as exc:
            raise OidcError("jwks", "Could not validate identity token signature") from exc

    def _discover(self) -> dict:
        if self._discovery is not None:
            return self._discovery
        url = settings.oidc_discovery_url.strip() or f"{self.issuer}/.well-known/openid-configuration"
        try:
            response = httpx.get(url, timeout=_HTTP_TIMEOUT)
            response.raise_for_status()
            document = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OidcError("discovery", "OIDC discovery failed") from exc
        discovered_issuer = str(document.get("issuer") or "").rstrip("/")
        if discovered_issuer != self.issuer:
            raise OidcError("discovery", "OIDC discovery failed")
        for key in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            if not document.get(key):
                raise OidcError("discovery", "OIDC discovery failed")
        self._discovery = document
        return document


def get_oidc_provider() -> OidcProvider:
    if not settings.oidc_enabled:
        raise OidcDisabledError()
    if not oidc_is_configured():
        raise AuthConfigError("OIDC is enabled but issuer, client id, or redirect URI is missing")
    return HttpOidcProvider()
