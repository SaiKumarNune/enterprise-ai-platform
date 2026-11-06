from __future__ import annotations

import json
import threading
from base64 import urlsafe_b64encode
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.oidc import get_oidc_provider
from app.core.config import settings
from app.core.security import hash_secret
from app.main import app
from app.models.identity import ExternalIdentity
from app.models.invitation import Invitation
from app.services.invitations import InvitationError, consume_invitation
from app.services.oidc import validate_id_token
from tests.auth_helpers import bearer_headers, seed_tenant, seed_user
from tests.fake_oidc import FakeOidcProvider, generate_rsa_key
from tests.test_documents import FakeSession, _clear_overrides, _override_db

client = TestClient(app)
PASSWORD = "correct-horse-battery"
ISSUER = "https://idp.test"
CLIENT_ID = "enterprise-ai"
REDIRECT = "http://127.0.0.1:8000/auth/oidc/callback"


@pytest.fixture(scope="module")
def rsa_key():
    return generate_rsa_key()


@pytest.fixture(scope="module")
def other_rsa_key():
    return generate_rsa_key()


def _patch_settings(monkeypatch, **changes):
    updated = replace(settings, **changes)
    monkeypatch.setattr("app.core.config.settings", updated)
    monkeypatch.setattr("app.api.auth.settings", updated)
    monkeypatch.setattr("app.api.oidc.settings", updated)
    monkeypatch.setattr("app.services.oidc.settings", updated)
    monkeypatch.setattr("app.services.users.settings", updated)
    monkeypatch.setattr("app.core.security.settings", updated)
    return updated


def _enable_oidc(monkeypatch, provider: FakeOidcProvider, *, invite_only: bool = True):
    _patch_settings(
        monkeypatch,
        oidc_enabled=True,
        oidc_issuer=ISSUER,
        oidc_client_id=CLIENT_ID,
        oidc_redirect_uri=REDIRECT,
        auth_invite_only=invite_only,
    )
    app.dependency_overrides[get_oidc_provider] = lambda: provider


def _start_login(session: FakeSession, provider: FakeOidcProvider) -> tuple[str, str]:
    _override_db(session)
    response = client.get("/auth/oidc/login", follow_redirects=False)
    assert response.status_code == 302
    params = parse_qs(urlparse(response.headers["location"]).query)
    assert params["code_challenge_method"] == ["S256"]
    assert params["response_type"] == ["code"]
    state = params["state"][0]
    nonce = params["nonce"][0]
    return state, nonce


def test_oidc_disabled_returns_404() -> None:
    session = FakeSession()
    _override_db(session)
    try:
        response = client.get("/auth/oidc/login", follow_redirects=False)
    finally:
        _clear_overrides()
    assert response.status_code == 404


def test_invite_only_blocks_register_without_invitation(monkeypatch) -> None:
    _patch_settings(monkeypatch, auth_invite_only=True)
    session = FakeSession()
    _override_db(session)
    try:
        response = client.post(
            "/auth/register",
            json={"email": "new@example.com", "password": PASSWORD, "role": "admin"},
        )
    finally:
        _clear_overrides()
    assert response.status_code == 403
    assert response.json()["detail"] == "Registration requires an invitation"


def test_invite_register_stays_employee_and_password_login_works(monkeypatch) -> None:
    _patch_settings(monkeypatch, auth_invite_only=True)
    session = FakeSession()
    admin = seed_user(session, role="admin", email="admin-invite@example.com")
    _override_db(session)
    try:
        created = client.post(
            "/admin/invitations",
            json={"email": "Invited@Example.com", "expires_in_hours": 24},
            headers=bearer_headers(admin),
        )
        token = created.json()["token"]
        registered = client.post(
            "/auth/register",
            json={
                "email": "invited@example.com",
                "password": PASSWORD,
                "role": "admin",
                "invite_token": token,
            },
        )
        login = client.post(
            "/auth/login",
            json={"email": "invited@example.com", "password": PASSWORD},
        )
        employee_invite = client.post(
            "/admin/invitations",
            json={"email": "other@example.com"},
            headers=bearer_headers(session.users[-1]),
        )
        reused = client.post(
            "/auth/register",
            json={
                "email": "invited@example.com",
                "password": PASSWORD,
                "invite_token": token,
            },
        )
    finally:
        _clear_overrides()
    assert created.status_code == 201
    assert created.json()["tenant_id"] == str(admin.tenant_id)
    assert registered.status_code == 201
    assert registered.json()["role"] == "employee"
    assert registered.json()["tenant_id"] == str(admin.tenant_id)
    assert login.status_code == 200
    assert employee_invite.status_code == 403
    assert reused.status_code in {403, 409}


def test_expired_invitation_cannot_register(monkeypatch) -> None:
    _patch_settings(monkeypatch, auth_invite_only=True)
    session = FakeSession()
    admin = seed_user(session, role="admin", email="exp-admin@example.com")
    session.add(
        Invitation(
            id=uuid4(),
            tenant_id=admin.tenant_id,
            email="late@example.com",
            token_hash=hash_secret("expired-invite-token"),
            created_by=admin.id,
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
    )
    _override_db(session)
    try:
        response = client.post(
            "/auth/register",
            json={
                "email": "late@example.com",
                "password": PASSWORD,
                "invite_token": "expired-invite-token",
            },
        )
    finally:
        _clear_overrides()
    assert response.status_code == 403
    assert session.users[-1].email != "late@example.com" or session.users[-1].id == admin.id


def test_invitation_token_is_single_use(monkeypatch) -> None:
    _patch_settings(monkeypatch, auth_invite_only=True)
    session = FakeSession()
    admin = seed_user(session, role="admin", email="once-admin@example.com")
    _override_db(session)
    try:
        created = client.post(
            "/admin/invitations",
            json={"email": "once@example.com"},
            headers=bearer_headers(admin),
        )
        token = created.json()["token"]
        first = client.post(
            "/auth/register",
            json={"email": "once@example.com", "password": PASSWORD, "invite_token": token},
        )
        second = client.post(
            "/auth/register",
            json={"email": "once@example.com", "password": PASSWORD, "invite_token": token},
        )
    finally:
        _clear_overrides()
    assert first.status_code == 201
    assert second.status_code in {403, 409}
    assert sum(1 for item in session.users if item.email == "once@example.com") == 1


def test_concurrent_invitation_redemption_claims_once() -> None:
    session = FakeSession()
    admin = seed_user(session, role="admin", email="race-admin@example.com")
    token = "concurrent-invite-token"
    session.add(
        Invitation(
            id=uuid4(),
            tenant_id=admin.tenant_id,
            email="race@example.com",
            token_hash=hash_secret(token),
            created_by=admin.id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            created_at=datetime.now(timezone.utc),
        )
    )
    barrier = threading.Barrier(2)
    winners: list[object] = []
    losers: list[object] = []

    def redeem() -> None:
        barrier.wait()
        try:
            winners.append(consume_invitation(session, email="race@example.com", token=token, commit=True))
        except InvitationError as exc:
            session.rollback()
            losers.append(exc)

    threads = [threading.Thread(target=redeem) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(winners) == 1
    assert len(losers) == 1
    assert session.invitations[0].consumed_at is not None


def test_existing_password_login_survives_invite_only(monkeypatch) -> None:
    _patch_settings(monkeypatch, auth_invite_only=True)
    session = FakeSession()
    seed_user(session, email="localdev@example.com", password=PASSWORD)
    _override_db(session)
    try:
        response = client.post(
            "/auth/login",
            json={"email": "localdev@example.com", "password": PASSWORD},
        )
    finally:
        _clear_overrides()
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"


def test_oidc_valid_login_issues_local_jwt(monkeypatch, rsa_key) -> None:
    provider = FakeOidcProvider(rsa_key, issuer=ISSUER, client_id=CLIENT_ID, redirect_uri=REDIRECT)
    _enable_oidc(monkeypatch, provider)
    session = FakeSession()
    admin = seed_user(session, role="admin", email="oidc-admin@example.com")
    invitation = Invitation(
        id=uuid4(),
        tenant_id=admin.tenant_id,
        email="oidc.user@example.com",
        token_hash=hash_secret("unused-invite-token"),
        created_by=admin.id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        created_at=datetime.now(timezone.utc),
    )
    session.add(invitation)
    try:
        state, nonce = _start_login(session, provider)
        code = provider.issue_code(
            nonce=nonce,
            email="oidc.user@example.com",
            sub="idp-user-1",
            extra={"roles": ["admin"], "groups": ["admins"], "role": "admin"},
        )
        callback = client.get("/auth/oidc/callback", params={"code": code, "state": state})
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {callback.json()['access_token']}"})
    finally:
        _clear_overrides()
    assert callback.status_code == 200
    assert me.status_code == 200
    assert me.json()["email"] == "oidc.user@example.com"
    assert me.json()["role"] == "employee"
    assert me.json()["tenant_id"] == str(admin.tenant_id)
    assert invitation.consumed_at is not None
    assert any(isinstance(item, ExternalIdentity) for item in session.identities)


def test_oidc_missing_invitation_rejected(monkeypatch, rsa_key) -> None:
    provider = FakeOidcProvider(rsa_key, issuer=ISSUER, client_id=CLIENT_ID, redirect_uri=REDIRECT)
    _enable_oidc(monkeypatch, provider)
    session = FakeSession()
    try:
        state, nonce = _start_login(session, provider)
        code = provider.issue_code(nonce=nonce, email="stranger@example.com", sub="idp-stranger")
        callback = client.get("/auth/oidc/callback", params={"code": code, "state": state})
    finally:
        _clear_overrides()
    assert callback.status_code == 403
    assert callback.json()["detail"] == "Registration requires an invitation"
    assert session.users == []


def test_oidc_invalid_and_expired_tokens(monkeypatch, rsa_key, other_rsa_key) -> None:
    provider = FakeOidcProvider(rsa_key, issuer=ISSUER, client_id=CLIENT_ID, redirect_uri=REDIRECT)
    _enable_oidc(monkeypatch, provider)
    session = FakeSession()
    try:
        state, nonce = _start_login(session, provider)
        bad_sig = provider.issue_code(
            nonce=nonce,
            email="sig@example.com",
            sub="sig-user",
            key=other_rsa_key,
        )
        invalid = client.get("/auth/oidc/callback", params={"code": bad_sig, "state": state})

        state, nonce = _start_login(session, provider)
        expired = provider.issue_code(
            nonce=nonce,
            email="exp@example.com",
            sub="exp-user",
            exp_delta=-120,
        )
        expired_resp = client.get("/auth/oidc/callback", params={"code": expired, "state": state})
    finally:
        _clear_overrides()
    assert invalid.status_code == 401
    assert expired_resp.status_code == 401


def test_validate_id_token_rejects_wrong_issuer_audience_nonce_and_alg_none(rsa_key) -> None:
    provider = FakeOidcProvider(rsa_key, issuer=ISSUER, client_id=CLIENT_ID, redirect_uri=REDIRECT)
    nonce = "nonce-1"
    valid = provider.encode_id_token(nonce=nonce, email="claims@example.com", sub="sub-1")
    claims = validate_id_token(
        valid,
        issuer=ISSUER,
        audience=CLIENT_ID,
        nonce=nonce,
        key=provider.public_key,
    )
    assert claims["email"] == "claims@example.com"

    expired = provider.encode_id_token(
        nonce=nonce, email="claims@example.com", sub="sub-1", exp_delta=-120,
    )
    with pytest.raises(Exception) as expired_exc:
        validate_id_token(expired, issuer=ISSUER, audience=CLIENT_ID, nonce=nonce, key=provider.public_key)
    assert expired_exc.value.code == "expired"

    wrong_aud = provider.encode_id_token(
        nonce=nonce, email="claims@example.com", sub="sub-1", audience="other-client"
    )
    with pytest.raises(Exception) as aud_exc:
        validate_id_token(wrong_aud, issuer=ISSUER, audience=CLIENT_ID, nonce=nonce, key=provider.public_key)
    assert aud_exc.value.code == "invalid_token"

    wrong_iss = provider.encode_id_token(
        nonce=nonce, email="claims@example.com", sub="sub-1", issuer="https://evil.test"
    )
    with pytest.raises(Exception) as iss_exc:
        validate_id_token(wrong_iss, issuer=ISSUER, audience=CLIENT_ID, nonce=nonce, key=provider.public_key)
    assert iss_exc.value.code == "invalid_token"

    with pytest.raises(Exception) as nonce_exc:
        validate_id_token(valid, issuer=ISSUER, audience=CLIENT_ID, nonce="other-nonce", key=provider.public_key)
    assert nonce_exc.value.code == "invalid_nonce"

    def _b64(value: dict) -> str:
        return urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=").decode()

    none_token = (
        f"{_b64({'alg': 'none', 'typ': 'JWT'})}."
        f"{_b64({'iss': ISSUER, 'aud': CLIENT_ID, 'sub': 'sub-1', 'email': 'claims@example.com', 'iat': 1, 'exp': 2000000000, 'nonce': nonce})}."
    )
    with pytest.raises(Exception):
        validate_id_token(none_token, issuer=ISSUER, audience=CLIENT_ID, nonce=nonce, key=provider.public_key)


def test_oidc_keeps_database_role_and_blocks_cross_tenant_remap(monkeypatch, rsa_key) -> None:
    provider = FakeOidcProvider(rsa_key, issuer=ISSUER, client_id=CLIENT_ID, redirect_uri=REDIRECT)
    _enable_oidc(monkeypatch, provider)
    session = FakeSession()
    tenant_a = seed_tenant(session, name="TenantA")
    tenant_b = seed_tenant(session, name="TenantB")
    manager = seed_user(
        session,
        role="manager",
        email="manager@example.com",
        tenant_id=tenant_a.id,
    )
    admin_b = seed_user(session, role="admin", email="admin-b@example.com", tenant_id=tenant_b.id)
    session.add(
        ExternalIdentity(
            id=uuid4(),
            user_id=manager.id,
            issuer=ISSUER,
            subject="stable-sub",
            email="manager@example.com",
        )
    )
    invite_b = Invitation(
        id=uuid4(),
        tenant_id=tenant_b.id,
        email="moved@example.com",
        token_hash=hash_secret("tenant-b-invite"),
        created_by=admin_b.id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        created_at=datetime.now(timezone.utc),
    )
    session.add(invite_b)
    try:
        state, nonce = _start_login(session, provider)
        code = provider.issue_code(
            nonce=nonce,
            email="moved@example.com",
            sub="stable-sub",
            extra={"role": "admin"},
        )
        callback = client.get("/auth/oidc/callback", params={"code": code, "state": state})
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {callback.json()['access_token']}"})
    finally:
        _clear_overrides()
    assert callback.status_code == 200
    assert me.json()["id"] == str(manager.id)
    assert me.json()["role"] == "manager"
    assert me.json()["tenant_id"] == str(tenant_a.id)
    assert invite_b.consumed_at is None
    assert len(session.users) == 2
    assert manager.tenant_id == tenant_a.id
    assert manager.role == "manager"
    assert session.identities[0].user_id == manager.id


def test_oidc_does_not_link_existing_password_account_by_email(monkeypatch, rsa_key) -> None:
    provider = FakeOidcProvider(rsa_key, issuer=ISSUER, client_id=CLIENT_ID, redirect_uri=REDIRECT)
    _enable_oidc(monkeypatch, provider, invite_only=False)
    session = FakeSession()
    local = seed_user(
        session,
        role="employee",
        email="local-owner@example.com",
        password=PASSWORD,
    )
    original_id = local.id
    original_tenant = local.tenant_id
    try:
        state, nonce = _start_login(session, provider)
        code = provider.issue_code(
            nonce=nonce,
            email="local-owner@example.com",
            sub="attacker-sub",
            extra={"role": "admin", "groups": ["admin"]},
        )
        callback = client.get("/auth/oidc/callback", params={"code": code, "state": state})
        password_login = client.post(
            "/auth/login",
            json={"email": "local-owner@example.com", "password": PASSWORD},
        )
    finally:
        _clear_overrides()
    assert callback.status_code == 409
    assert password_login.status_code == 200
    assert local.id == original_id
    assert local.tenant_id == original_tenant
    assert local.role == "employee"
    assert session.identities == []
    assert len(session.users) == 1
