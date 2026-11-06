from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from mcp.shared.memory import create_connected_server_and_client_session

from app.core.config import settings
from app.main import app
from app.mcp.client import invoke_create_ticket, invoke_prepare_ticket
from app.mcp.context import McpExecutionContext
from app.mcp.server import build_mcp_server
from app.services.agent_tools import ToolError, dispatch_tool, parse_planner_output
from app.services.authorization import Principal
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from app.services.ticketing import (
    MockTicketAdapter,
    TicketingDisabledError,
    get_ticket_adapter,
    reset_ticket_adapter,
)
from tests.auth_helpers import bearer_headers, seed_agent_users, seed_tenant, seed_user
from tests.fake_embeddings import HashEmbedder
from tests.fake_llm import SequenceLLM
from tests.test_agents import _session_with_policy
from tests.test_documents import FakeSession, _clear_overrides, _override_db

client = TestClient(app)

PLAN_TICKET = '{"intent":"draft_action","tool":"prepare_ticket","action_type":"create_ticket"}'
TICKET_DRAFT = (
    '{"action_type":"create_ticket","title":"VPN access is broken",'
    '"description":"Open a ticket because VPN access is broken","priority":"high","source_ids":[]}'
)
TICKET_REQUEST = "Open a ticket because VPN access is broken"


@pytest.fixture(autouse=True)
def _reset_tickets():
    reset_ticket_adapter()
    yield
    reset_ticket_adapter()


def _enable_ticketing(monkeypatch) -> None:
    updated = replace(settings, ticketing_enabled=True)
    monkeypatch.setattr("app.core.config.settings", updated)
    monkeypatch.setattr("app.mcp.server.settings", updated)
    monkeypatch.setattr("app.services.ticketing.settings", updated)


def _override_llm() -> None:
    app.dependency_overrides[get_llm] = lambda: SequenceLLM([PLAN_TICKET, TICKET_DRAFT])
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()


def _context(user, *, authorized: bool = False, workflow_id=None):
    return McpExecutionContext(
        principal=Principal.from_user(user),
        execution_authorized=authorized,
        workflow_id=workflow_id,
    )


def _call(context, name: str, arguments: dict):
    async def _run():
        server = build_mcp_server(context)
        async with create_connected_server_and_client_session(server) as session:
            return await session.call_tool(name, arguments)

    return asyncio.run(_run())


def test_ticketing_disabled_by_default() -> None:
    session = FakeSession()
    user = seed_user(session)
    with pytest.raises(TicketingDisabledError):
        get_ticket_adapter()
    denied = _call(
        _context(user, authorized=True, workflow_id=uuid4()),
        "create_ticket",
        {"title": "VPN access is broken", "description": "VPN down", "priority": "high"},
    )
    assert denied.isError is True
    assert "disabled" in denied.content[0].text.lower()


def test_planner_routes_ticket_requests_without_breaking_leave() -> None:
    ticket = parse_planner_output("not json", TICKET_REQUEST)
    leave = parse_planner_output(
        "not json",
        "Prepare an approval request for three days off.",
    )
    assert ticket["action_type"] == "create_ticket"
    assert ticket["tool"] == "prepare_ticket"
    assert leave["action_type"] == "leave_request"


def test_prepare_ticket_strips_identity_and_does_not_create(monkeypatch) -> None:
    _enable_ticketing(monkeypatch)
    session = FakeSession()
    user = seed_user(session)
    draft = invoke_prepare_ticket(
        principal=Principal.from_user(user),
        request=TICKET_REQUEST,
        raw_draft='{"action_type":"create_ticket","title":"VPN access is broken","description":"VPN down","priority":"high","tenant_id":"other","user_id":"attacker","authorized":true}',
        valid_source_ids=set(),
    )
    assert draft is not None
    assert draft["action_type"] == "create_ticket"
    assert draft["requires_approval"] is True
    assert "tenant_id" not in draft
    assert "user_id" not in draft
    assert "authorized" not in draft
    adapter = get_ticket_adapter()
    assert isinstance(adapter, MockTicketAdapter)
    assert adapter.create_calls == 0


def test_create_ticket_mcp_requires_approval_and_rejects_identity(monkeypatch) -> None:
    _enable_ticketing(monkeypatch)
    session = FakeSession()
    user = seed_user(session)
    workflow_id = uuid4()
    denied = _call(
        _context(user, authorized=False, workflow_id=workflow_id),
        "create_ticket",
        {"title": "VPN access is broken", "description": "VPN down", "priority": "high"},
    )
    leaked = _call(
        _context(user, authorized=True, workflow_id=workflow_id),
        "create_ticket",
        {
            "title": "VPN access is broken",
            "description": "VPN down",
            "priority": "high",
            "tenant_id": str(uuid4()),
            "user_id": str(uuid4()),
            "authorized": True,
        },
    )
    assert denied.isError is True
    assert "approval" in denied.content[0].text.lower()
    assert leaked.isError is True
    assert "identity" in leaked.content[0].text.lower()
    with pytest.raises(ToolError):
        dispatch_tool("create_ticket", title="VPN", description="down", priority="high")


def test_ticket_approval_creates_once_and_reject_does_not(monkeypatch) -> None:
    _enable_ticketing(monkeypatch)
    session, _document = _session_with_policy()
    _override_llm()
    _override_db(session)
    emp, mgr, _admin = seed_agent_users(session)
    try:
        started = client.post(
            "/agents/run",
            json={"request": TICKET_REQUEST},
            headers=emp,
        )
        workflow_id = started.json()["workflow_id"]
        rejected_id = client.post(
            "/agents/run",
            json={"request": TICKET_REQUEST},
            headers=emp,
        ).json()["workflow_id"]
        rejected = client.post(f"/agents/{rejected_id}/reject", json={}, headers=mgr)
        approved = client.post(f"/agents/{workflow_id}/approve", json={}, headers=mgr)
        duplicate = client.post(f"/agents/{workflow_id}/approve", json={}, headers=mgr)
    finally:
        _clear_overrides()

    assert started.status_code == 200
    assert started.json()["status"] == "pending_approval"
    assert started.json()["draft"]["action_type"] == "create_ticket"
    assert started.json()["execution_result"] is None
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert rejected.json()["execution_result"] is None
    assert approved.status_code == 200
    result = approved.json()["execution_result"]
    assert result["created"] is True
    assert result["ticket_id"]
    assert result["tenant_id"]
    assert duplicate.status_code == 409
    adapter = get_ticket_adapter()
    assert adapter.create_calls == 1


def test_ticket_cross_tenant_approve_is_not_found(monkeypatch) -> None:
    _enable_ticketing(monkeypatch)
    session, _document = _session_with_policy()
    _override_llm()
    _override_db(session)
    emp, _mgr, _admin = seed_agent_users(session)
    outsider = seed_user(session, role="manager", tenant_id=seed_tenant(session, name="OtherCo").id)
    try:
        started = client.post("/agents/run", json={"request": TICKET_REQUEST}, headers=emp)
        blocked = client.post(
            f"/agents/{started.json()['workflow_id']}/approve",
            json={},
            headers=bearer_headers(outsider),
        )
    finally:
        _clear_overrides()
    assert started.status_code == 200
    assert blocked.status_code == 404


def test_create_ticket_is_idempotent_and_bound_to_context_tenant(monkeypatch) -> None:
    _enable_ticketing(monkeypatch)
    session = FakeSession()
    tenant_a = seed_tenant(session, name="A")
    tenant_b = seed_tenant(session, name="B")
    user_a = seed_user(session, tenant_id=tenant_a.id)
    user_b = seed_user(session, tenant_id=tenant_b.id)
    workflow_id = uuid4()
    first = invoke_create_ticket(
        principal=Principal.from_user(user_a),
        title="VPN access is broken",
        description="VPN down",
        priority="high",
        workflow_id=workflow_id,
    )
    second = invoke_create_ticket(
        principal=Principal.from_user(user_a),
        title="VPN access is broken",
        description="VPN down",
        priority="high",
        workflow_id=workflow_id,
    )
    other = invoke_create_ticket(
        principal=Principal.from_user(user_b),
        title="VPN access is broken",
        description="VPN down",
        priority="high",
        workflow_id=workflow_id,
    )
    assert first["ticket_id"] == second["ticket_id"]
    assert first["tenant_id"] == str(tenant_a.id)
    assert other["tenant_id"] == str(tenant_b.id)
    assert other["ticket_id"] != first["ticket_id"]
    adapter = get_ticket_adapter()
    assert adapter.create_calls == 2


def test_leave_request_still_simulates_when_ticketing_enabled(monkeypatch) -> None:
    from tests.test_agents import DRAFT, PLAN_DRAFT

    _enable_ticketing(monkeypatch)
    session, _document = _session_with_policy()
    app.dependency_overrides[get_llm] = lambda: SequenceLLM([PLAN_DRAFT, DRAFT])
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    _override_db(session)
    emp, mgr, _admin = seed_agent_users(session)
    try:
        started = client.post(
            "/agents/run",
            json={"request": "Prepare an approval request for three days off."},
            headers=emp,
        )
        approved = client.post(
            f"/agents/{started.json()['workflow_id']}/approve",
            json={},
            headers=mgr,
        )
    finally:
        _clear_overrides()
    assert started.json()["draft"]["action_type"] == "leave_request"
    assert approved.json()["execution_result"]["simulated"] is True
    assert "ticket_id" not in approved.json()["execution_result"]
    assert get_ticket_adapter().create_calls == 0
