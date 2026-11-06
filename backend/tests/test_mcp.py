import asyncio
import uuid

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from app.mcp.client import invoke_knowledge_search, invoke_prepare_approval_request
from app.mcp.context import McpExecutionContext
from app.mcp.errors import McpToolError
from app.mcp.server import EXPOSED_TOOLS, build_mcp_server
from app.services.agent_tools import execute_approved_action
from app.services.agents.action import prepare_leave_draft
from app.services.agents.research import run_research
from app.services.authorization import DocumentAccessDenied, Principal
from app.services.agent_tools import ToolError
from tests.auth_helpers import own_session_documents, seed_tenant, seed_user
from tests.fake_embeddings import HashEmbedder
from tests.fake_llm import SequenceLLM
from tests.test_agents import _session_with_policy
from tests.test_documents import FakeSession


def _context_for(user, session=None, embedder=None) -> McpExecutionContext:
    return McpExecutionContext(
        principal=Principal.from_user(user),
        db=session,
        embedder=embedder,
    )


def _call(context: McpExecutionContext, name: str, arguments: dict):
    async def _run():
        server = build_mcp_server(context)
        async with create_connected_server_and_client_session(server) as session:
            return await session.call_tool(name, arguments)

    return asyncio.run(_run())


def test_mcp_lists_only_non_executing_tools() -> None:
    session = FakeSession()
    user = seed_user(session)

    async def _run():
        server = build_mcp_server(_context_for(user, session, HashEmbedder()))
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            return [tool.name for tool in listed.tools]

    names = asyncio.run(_run())
    assert set(names) == EXPOSED_TOOLS
    assert "execute_approved_action" not in names


def test_mcp_knowledge_search_returns_authorized_excerpts() -> None:
    session, document = _session_with_policy()
    user = seed_user(session)
    own_session_documents(session, user)
    result = invoke_knowledge_search(
        db=session,
        embedder=HashEmbedder(),
        principal=Principal.from_user(user),
        question="How many vacation days do full-time employees receive?",
        top_k=5,
        document_id=document.id,
    )
    assert result["selected_count"] == 1
    assert result["retrieval_count"] >= 1
    assert result["excerpts"][0]["source_id"] == "S1"
    assert "vacation days" in result["excerpts"][0]["content"]
    assert result["sources"][0]["document_id"] == str(document.id)


def test_mcp_prepare_approval_request_does_not_execute() -> None:
    session = FakeSession()
    user = seed_user(session)
    payload = invoke_prepare_approval_request(
        principal=Principal.from_user(user),
        request="Prepare an approval request for three days off because a family visit from 2026-11-03 to 2026-11-05.",
        raw_draft='{"action_type":"leave_request","days":3,"source_ids":["S1"]}',
        valid_source_ids={"S1"},
        action_type="leave_request",
    )
    assert payload is not None
    assert payload["action_type"] == "leave_request"
    assert payload["requires_approval"] is True
    assert payload["days"] == 3
    assert "simulated" not in payload


def test_mcp_schema_validation_rejects_blank_question() -> None:
    session, document = _session_with_policy()
    user = seed_user(session)
    own_session_documents(session, user)
    result = _call(
        _context_for(user, session, HashEmbedder()),
        "knowledge_search",
        {"question": "   ", "document_id": str(document.id)},
    )
    assert result.isError is True


def test_mcp_rejects_agent_supplied_identity_fields() -> None:
    session, document = _session_with_policy()
    user = seed_user(session)
    own_session_documents(session, user)
    outsider = seed_user(session, tenant_id=seed_tenant(session, name="Other").id)
    result = _call(
        _context_for(user, session, HashEmbedder()),
        "knowledge_search",
        {
            "question": "vacation days",
            "document_id": str(document.id),
            "user_id": str(outsider.id),
            "tenant_id": str(outsider.tenant_id),
            "role": "admin",
            "authorized": True,
        },
    )
    assert result.isError is True
    assert "identity" in result.content[0].text.lower()


def test_mcp_unauthorized_inactive_user_cannot_search() -> None:
    session, document = _session_with_policy()
    user = seed_user(session, is_active=False)
    own_session_documents(session, user)
    with pytest.raises(McpToolError):
        invoke_knowledge_search(
            db=session,
            embedder=HashEmbedder(),
            principal=Principal.from_user(user),
            question="vacation days",
            top_k=5,
            document_id=document.id,
        )


def test_mcp_cross_tenant_document_is_not_found() -> None:
    session, document = _session_with_policy()
    owner = seed_user(session)
    own_session_documents(session, owner)
    outsider = seed_user(session, tenant_id=seed_tenant(session, name="OtherCo").id)
    with pytest.raises(McpToolError, match="not found"):
        invoke_knowledge_search(
            db=session,
            embedder=HashEmbedder(),
            principal=Principal.from_user(outsider),
            question="vacation days",
            top_k=5,
            document_id=document.id,
        )


def test_mcp_does_not_expose_execute_approved_action() -> None:
    session = FakeSession()
    user = seed_user(session)
    result = _call(
        _context_for(user),
        "execute_approved_action",
        {"draft": {"action_type": "leave_request"}, "authorized": True},
    )
    assert result.isError is True
    with pytest.raises(ToolError):
        execute_approved_action(draft={"action_type": "leave_request"}, authorized=False)


def test_research_agent_invokes_mcp_knowledge_search() -> None:
    session, document = _session_with_policy()
    user = seed_user(session)
    own_session_documents(session, user)
    result = run_research(
        session,
        HashEmbedder(),
        question="How many vacation days do full-time employees receive?",
        top_k=5,
        document_id=str(document.id),
        principal=Principal.from_user(user),
    )
    assert result["selected_count"] == 1
    assert result["sources"][0]["chunk_id"]


def test_action_agent_invokes_mcp_prepare_without_executing() -> None:
    session = FakeSession()
    user = seed_user(session)
    llm = SequenceLLM(
        ['{"reason":"personal reasons","start_date":"2026-11-03","days":3,"source_ids":["S1"]}']
    )
    draft = prepare_leave_draft(
        llm,
        request="Prepare an approval request for three days off.",
        excerpts=[{"source_id": "S1", "content": "Employees receive 15 vacation days."}],
        sources=[{"source_id": "S1", "document_id": str(uuid.uuid4()), "chunk_id": "c", "page_number": 1}],
        action_type="leave_request",
        principal=Principal.from_user(user),
    )
    assert draft is not None
    assert draft["requires_approval"] is True
    assert "simulated" not in draft


def test_run_research_maps_cross_tenant_mcp_errors() -> None:
    session, document = _session_with_policy()
    owner = seed_user(session)
    own_session_documents(session, owner)
    outsider = seed_user(session, tenant_id=seed_tenant(session, name="Isle").id)
    with pytest.raises(DocumentAccessDenied):
        run_research(
            session,
            HashEmbedder(),
            question="vacation days",
            top_k=5,
            document_id=str(document.id),
            principal=Principal.from_user(outsider),
        )
