from __future__ import annotations

import json
import re
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.services.embeddings import EmbeddingBackend
from app.services.rag import retrieve_evidence, rewrite_citations

ALLOWED_TOOLS = frozenset(
    {"knowledge_search", "prepare_approval_request", "execute_approved_action"}
)
ALLOWED_INTENTS = frozenset({"informational", "draft_action", "clarification"})
ALLOWED_ACTION_TYPES = frozenset({"leave_request"})
PLANNER_TOOLS = frozenset({"knowledge_search", "prepare_approval_request", "none", None})

LEAVE_HINTS = (
    "approval request",
    "days off",
    "leave request",
    "time off",
    "vacation request",
)
DAYS_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})\s+(?:vacation\s+|calendar\s+)?days?\b",
    re.IGNORECASE,
)
WORD_DAYS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
BECAUSE_RE = re.compile(
    r"\bbecause\s+(.+?)(?=\s+from\s+\d{4}-\d{2}-\d{2}|\s+starting\b|\s+on\s+\d{4}-\d{2}-\d{2}|$)",
    re.IGNORECASE | re.DOTALL,
)
NOT_PROVIDED = "Not provided"

SIMULATED_EXECUTION_MESSAGE = (
    "Recorded a simulated leave_request. No email was sent, no document was "
    "changed, and no external system was contacted."
)


class ToolError(ValueError):
    """Invalid tool name or arguments."""


def parse_planner_output(raw: str, request: str) -> dict[str, Any]:
    payload = _extract_json(raw)
    if payload is None:
        return _fallback_plan(request, reason="unparseable_planner_output")

    intent = payload.get("intent")
    tool = payload.get("tool")
    if tool in {"", "null"}:
        tool = None
    action_type = payload.get("action_type")
    if action_type in {"", "null", None}:
        action_type = None

    if intent not in ALLOWED_INTENTS:
        return _fallback_plan(request, reason="unknown_intent")
    if tool not in PLANNER_TOOLS:
        return _fallback_plan(request, reason="unknown_or_forbidden_tool")
    if action_type is not None and action_type not in ALLOWED_ACTION_TYPES:
        return _fallback_plan(request, reason="unknown_action_type")
    if intent != "draft_action":
        action_type = None
    elif action_type is None:
        action_type = "leave_request"
    return {
        "intent": intent,
        "tool": tool or "knowledge_search",
        "action_type": action_type,
        "planner_fallback": False,
    }


def knowledge_search(
    db: Session,
    embedder: EmbeddingBackend,
    *,
    question: str,
    top_k: int,
    document_id: UUID | None,
) -> dict[str, Any]:
    selected, sources, retrieval_count = retrieve_evidence(
        db,
        embedder,
        question=question,
        top_k=top_k,
        document_id=document_id,
    )
    excerpts = [
        {"source_id": f"S{index}", "content": chunk.content}
        for index, (chunk, _score) in enumerate(selected, start=1)
    ]
    source_payload = [
        {
            "source_id": source.source_id,
            "document_id": str(source.document_id),
            "chunk_id": str(source.chunk_id),
            "chunk_index": source.chunk_index,
            "page_number": source.page_number,
            "similarity": source.similarity,
        }
        for source in sources
    ]
    return {
        "excerpts": excerpts,
        "sources": source_payload,
        "retrieval_count": retrieval_count,
        "selected": selected,
    }


def prepare_approval_request(
    *,
    request: str,
    raw_draft: str,
    valid_source_ids: set[str],
    action_type: str | None,
) -> dict[str, Any] | None:
    if action_type not in ALLOWED_ACTION_TYPES:
        return None
    payload = _extract_json(raw_draft) or {}
    days = _parse_days(request)
    if days is None or days < 1 or days > 30:
        return None
    provided = extract_user_leave_details(request)
    reason = _user_supplied_value(request, payload.get("reason")) or provided["reason"]
    start_date = _user_supplied_value(request, payload.get("start_date")) or provided["start_date"]
    end_date = _user_supplied_value(request, payload.get("end_date")) or provided["end_date"]
    employee_name = _user_supplied_value(request, payload.get("employee_name"))
    raw_details = " ".join(
        str(payload.get(key) or "")
        for key in ("details", "summary", "reason", "title")
    )
    _, cited_ids, had_invalid = rewrite_citations(raw_details, valid_source_ids)
    requested_ids = payload.get("source_ids") or cited_ids
    source_ids = []
    for label in requested_ids:
        if not isinstance(label, str):
            continue
        if label in valid_source_ids:
            if label not in source_ids:
                source_ids.append(label)
        else:
            had_invalid = True
    if not source_ids:
        source_ids = [label for label in sorted(valid_source_ids, key=lambda item: int(item[1:]))]
    details = _build_leave_details(
        days=days,
        reason=reason,
        start_date=start_date,
        end_date=end_date,
        source_ids=source_ids,
    )
    return {
        "action_type": "leave_request",
        "title": f"Leave request for {days} day(s)",
        "details": details,
        "days": days,
        "reason": reason or NOT_PROVIDED,
        "start_date": start_date or NOT_PROVIDED,
        "end_date": end_date or NOT_PROVIDED,
        "employee_name": employee_name or NOT_PROVIDED,
        "source_ids": source_ids,
        "requires_approval": True,
        "had_invalid_citations": had_invalid,
    }


def execute_approved_action(
    *,
    draft: dict[str, Any],
    authorized: bool,
) -> dict[str, Any]:
    if not authorized:
        raise ToolError("execute_approved_action requires an explicit API approval")
    if not draft or draft.get("action_type") not in ALLOWED_ACTION_TYPES:
        raise ToolError("draft is not an allowlisted action")
    return {
        "simulated": True,
        "action_type": draft["action_type"],
        "days": draft.get("days"),
        "message": SIMULATED_EXECUTION_MESSAGE,
    }


def dispatch_tool(name: str, **kwargs: Any) -> Any:
    if name not in ALLOWED_TOOLS:
        raise ToolError(f"unknown tool: {name}")
    if name == "knowledge_search":
        return knowledge_search(**kwargs)
    if name == "prepare_approval_request":
        return prepare_approval_request(**kwargs)
    if name == "execute_approved_action":
        return execute_approved_action(**kwargs)
    raise ToolError(f"unknown tool: {name}")


def _fallback_plan(request: str, *, reason: str) -> dict[str, Any]:
    lowered = request.lower()
    if any(hint in lowered for hint in LEAVE_HINTS):
        return {
            "intent": "draft_action",
            "tool": "knowledge_search",
            "action_type": "leave_request",
            "planner_fallback": True,
            "fallback_reason": reason,
        }
    return {
        "intent": "informational",
        "tool": "knowledge_search",
        "action_type": None,
        "planner_fallback": True,
        "fallback_reason": reason,
    }


def extract_user_leave_details(request: str) -> dict[str, Any]:
    because = BECAUSE_RE.search(request)
    reason = None
    if because:
        reason = " ".join(because.group(1).split()).strip(" .,")
        if not reason or (DAYS_RE.search(reason) and len(reason.split()) <= 5):
            reason = None
    dates = ISO_DATE_RE.findall(request)
    return {
        "days": parse_leave_days(request),
        "reason": reason,
        "start_date": dates[0] if dates else None,
        "end_date": dates[1] if len(dates) > 1 else None,
    }


def parse_leave_days(request: str) -> int | None:
    match = DAYS_RE.search(request)
    if match is None:
        return None
    token = match.group(1).lower()
    if token.isdigit():
        return int(token)
    return WORD_DAYS.get(token)


def _parse_days(request: str) -> int | None:
    return parse_leave_days(request)
    match = DAYS_RE.search(request)
    if match is None:
        return None
    token = match.group(1).lower()
    if token.isdigit():
        return int(token)
    return WORD_DAYS.get(token)


def _user_supplied_value(request: str, candidate: Any) -> str | None:
    if not isinstance(candidate, str):
        return None
    value = " ".join(candidate.split())
    if not value or value.lower() in {"not provided", "none", "null", "n/a", "unknown"}:
        return None
    if value.lower() not in request.lower():
        return None
    if DAYS_RE.search(value) and len(value.split()) <= 5:
        return None
    return value


def _build_leave_details(
    *,
    days: int,
    reason: str | None,
    start_date: str | None,
    end_date: str | None,
    source_ids: list[str],
) -> str:
    parts = [
        f"Leave request for {days} day(s).",
        f"Reason: {reason or NOT_PROVIDED}.",
        f"Start date: {start_date or NOT_PROVIDED}.",
        f"End date: {end_date or NOT_PROVIDED}.",
    ]
    if source_ids:
        citation = "[" + ", ".join(source_ids) + "]"
        parts.append(
            f"Policy excerpts cited {citation}. Annual entitlement is a policy limit, "
            "not a verified remaining balance."
        )
    return " ".join(parts)


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        match = JSON_RE.search(text)
        if match is None:
            return None
        try:
            loaded = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None


