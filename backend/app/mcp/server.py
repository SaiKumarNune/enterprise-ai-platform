"""Local MCP server. Tools reuse existing services; identity comes from context."""

from __future__ import annotations

from uuid import UUID

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.server import Settings as FastMCPSettings

# FastMCP Settings.lifespan annotates FastMCP before the class body finishes.
FastMCPSettings.model_rebuild()

from app.mcp.context import FORBIDDEN_IDENTITY_KEYS, McpExecutionContext
from app.mcp.errors import McpToolError
from app.mcp.schemas import (
    CreateTicketInput,
    CreateTicketOutput,
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
    PrepareApprovalInput,
    PrepareApprovalOutput,
    PrepareTicketInput,
    PrepareTicketOutput,
)
from app.core.config import settings
from app.services.agent_tools import knowledge_search, prepare_approval_request, prepare_ticket
from app.services.authorization import DocumentAccessDenied
from app.services.ticketing import (
    TicketingDisabledError,
    TicketingError,
    get_ticket_adapter,
    ticket_record_payload,
)

EXPOSED_TOOLS = frozenset(
    {"knowledge_search", "prepare_approval_request", "prepare_ticket", "create_ticket"}
)


def build_mcp_server(context: McpExecutionContext) -> FastMCP:
    server = FastMCP(
        "enterprise-ai-platform",
        instructions=(
            "Local tools for authorized knowledge search, leave-request drafts, "
            "and mock ticket drafts. Ticket creation is gated on API approval and "
            "the bound principal; identity arguments are ignored."
        ),
        log_level="ERROR",
    )

    @server.tool(name="knowledge_search", description="Retrieve authorized policy excerpts.")
    def knowledge_search_tool(
        question: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> KnowledgeSearchOutput:
        _require_active_principal(context)
        payload = KnowledgeSearchInput(question=question, top_k=top_k, document_id=document_id)
        if context.db is None or context.embedder is None:
            raise McpToolError("internal_error", "Search is unavailable")
        parsed_document_id = None
        if payload.document_id:
            try:
                parsed_document_id = UUID(payload.document_id)
            except ValueError as exc:
                raise McpToolError("invalid_arguments", "document_id must be a UUID") from exc
        try:
            result = knowledge_search(
                context.db,
                context.embedder,
                question=payload.question,
                top_k=payload.top_k,
                document_id=parsed_document_id,
                principal=context.principal,
            )
        except DocumentAccessDenied as exc:
            raise McpToolError("not_found", "Document not found") from exc
        except McpToolError:
            raise
        except Exception as exc:
            raise McpToolError("search_failed", "Knowledge search failed") from exc
        excerpts = result.get("excerpts") or []
        return KnowledgeSearchOutput(
            excerpts=excerpts,
            sources=result.get("sources") or [],
            retrieval_count=int(result.get("retrieval_count") or 0),
            selected_count=len(excerpts),
        )

    @server.tool(
        name="prepare_approval_request",
        description="Prepare a leave_request draft. Does not execute the request.",
    )
    def prepare_approval_request_tool(
        request: str,
        raw_draft: str,
        valid_source_ids: list[str] | None = None,
        action_type: str | None = "leave_request",
    ) -> PrepareApprovalOutput:
        _require_active_principal(context)
        payload = PrepareApprovalInput(
            request=request,
            raw_draft=raw_draft,
            valid_source_ids=list(valid_source_ids or []),
            action_type=action_type,
        )
        draft = prepare_approval_request(
            request=payload.request,
            raw_draft=payload.raw_draft,
            valid_source_ids=set(payload.valid_source_ids),
            action_type=payload.action_type,
        )
        return PrepareApprovalOutput(draft=draft)

    @server.tool(
        name="prepare_ticket",
        description="Prepare a mock helpdesk ticket draft. Does not create a ticket.",
    )
    def prepare_ticket_tool(
        request: str,
        raw_draft: str,
        valid_source_ids: list[str] | None = None,
    ) -> PrepareTicketOutput:
        _require_active_principal(context)
        payload = PrepareTicketInput(
            request=request,
            raw_draft=raw_draft,
            valid_source_ids=list(valid_source_ids or []),
        )
        draft = prepare_ticket(
            request=payload.request,
            raw_draft=payload.raw_draft,
            valid_source_ids=set(payload.valid_source_ids),
        )
        return PrepareTicketOutput(draft=draft)

    @server.tool(
        name="create_ticket",
        description="Create a mock helpdesk ticket after explicit API approval. Identity comes from context.",
    )
    def create_ticket_tool(
        title: str,
        description: str,
        priority: str = "medium",
    ) -> CreateTicketOutput:
        _require_active_principal(context)
        if not context.execution_authorized:
            raise McpToolError("forbidden", "Ticket creation requires an explicit API approval")
        if not settings.ticketing_enabled:
            raise McpToolError("disabled", "Ticketing is disabled")
        if context.workflow_id is None:
            raise McpToolError("invalid_arguments", "Ticket creation requires a bound workflow")
        payload = CreateTicketInput(title=title, description=description, priority=priority)
        try:
            record = get_ticket_adapter().create_ticket(
                tenant_id=context.principal.tenant_id,
                requester_user_id=context.principal.user_id,
                title=payload.title,
                description=payload.description,
                priority=payload.priority,
                idempotency_key=str(context.workflow_id),
            )
        except TicketingDisabledError as exc:
            raise McpToolError(exc.code, exc.message) from exc
        except TicketingError as exc:
            raise McpToolError(exc.code, exc.message) from exc
        return CreateTicketOutput(**ticket_record_payload(record))

    _install_argument_guard(server)
    return server


def _require_active_principal(context: McpExecutionContext) -> None:
    principal = context.principal
    if principal is None or not principal.is_active or principal.tenant_id is None:
        raise McpToolError("unauthorized", "Not authorized")


def _install_argument_guard(server: FastMCP) -> None:
    manager = server._tool_manager
    original = manager.call_tool

    async def call_tool(name: str, arguments: dict, context=None, convert_result: bool = False):
        if name not in EXPOSED_TOOLS:
            raise ToolError("Unknown tool")
        if not isinstance(arguments, dict):
            raise ToolError("invalid arguments")
        leaked = FORBIDDEN_IDENTITY_KEYS.intersection(arguments)
        if leaked:
            raise ToolError("Tool arguments must not include identity or authorization fields")
        try:
            return await original(name, arguments, context=context, convert_result=convert_result)
        except McpToolError as exc:
            raise ToolError(exc.message) from exc

    manager.call_tool = call_tool  # type: ignore[method-assign]
