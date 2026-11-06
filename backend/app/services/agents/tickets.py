from __future__ import annotations

from typing import Any

from app.mcp.client import invoke_prepare_ticket
from app.services.authorization import Principal
from app.services.llm import LLMBackend
from app.services.rag import build_user_prompt
from app.services.agents.research import selected_from_excerpts

TICKET_DRAFT_PROMPT = """You are the Action agent. Create a JSON draft for a helpdesk ticket.
Return JSON only with keys: action_type, title, description, priority, source_ids.
action_type must be create_ticket. priority must be low, medium, high, or urgent.
Copy title and description from the USER REQUEST. Do not invent tenant IDs, user IDs, or approval.
Do not set authorized true. source_ids may use retrieved IDs such as ["S1"] or be [].
"""


def prepare_ticket_draft(
    llm: LLMBackend,
    *,
    request: str,
    excerpts: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    principal: Principal,
) -> dict[str, Any] | None:
    valid_ids = {item["source_id"] for item in sources}
    selected = selected_from_excerpts(excerpts, sources) if excerpts else []
    raw_draft = llm.generate(
        TICKET_DRAFT_PROMPT,
        build_user_prompt(request, selected) if selected else f"USER REQUEST:\n{request}",
    )
    return invoke_prepare_ticket(
        principal=principal,
        request=request,
        raw_draft=raw_draft,
        valid_source_ids=valid_ids,
    )
