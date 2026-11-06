"""MCP client used by LangGraph Research and Action agents."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from uuid import UUID

from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from app.mcp.context import McpExecutionContext
from app.mcp.errors import McpToolError
from app.mcp.server import build_mcp_server
from app.services.authorization import Principal
from app.services.embeddings import EmbeddingBackend
from sqlalchemy.orm import Session


def invoke_knowledge_search(
    *,
    db: Session,
    embedder: EmbeddingBackend,
    principal: Principal,
    question: str,
    top_k: int,
    document_id: UUID | None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {"question": question, "top_k": top_k}
    if document_id is not None:
        arguments["document_id"] = str(document_id)
    payload = _call_tool(
        McpExecutionContext(principal=principal, db=db, embedder=embedder),
        "knowledge_search",
        arguments,
    )
    return payload


def invoke_prepare_approval_request(
    *,
    principal: Principal,
    request: str,
    raw_draft: str,
    valid_source_ids: set[str],
    action_type: str | None,
) -> dict[str, Any] | None:
    payload = _call_tool(
        McpExecutionContext(principal=principal),
        "prepare_approval_request",
        {
            "request": request,
            "raw_draft": raw_draft,
            "valid_source_ids": sorted(valid_source_ids),
            "action_type": action_type or "leave_request",
        },
    )
    return payload.get("draft")


def _call_tool(context: McpExecutionContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = _run_isolated(_invoke(context, name, arguments))
    if result.isError:
        raise McpToolError("tool_error", _error_text(result))
    structured = result.structuredContent
    if isinstance(structured, dict):
        return structured
    if result.content and isinstance(result.content[0], TextContent):
        parsed = json.loads(result.content[0].text)
        if isinstance(parsed, dict):
            return parsed
    raise McpToolError("tool_error", "MCP tool returned an invalid payload")


async def _invoke(context: McpExecutionContext, name: str, arguments: dict[str, Any]) -> CallToolResult:
    server = build_mcp_server(context)
    async with create_connected_server_and_client_session(server) as session:
        return await session.call_tool(name, arguments)


def _error_text(result: CallToolResult) -> str:
    if result.content and isinstance(result.content[0], TextContent):
        text = result.content[0].text.strip()
        if text:
            return text
    return "MCP tool failed"


def _run_isolated(coro):
    holder: list[Any] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            holder.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 — transport errors must surface
            errors.append(exc)

    thread = threading.Thread(target=worker, name="mcp-client")
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return holder[0]
