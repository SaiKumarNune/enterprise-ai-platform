"""Local ticketing adapter. No vendor account or paid API is required."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID, uuid4

from app.core.config import settings

ALLOWED_PRIORITIES = frozenset({"low", "medium", "high", "urgent"})
TICKET_HINTS = (
    "open a ticket",
    "create a ticket",
    "file a ticket",
    "helpdesk ticket",
    "support ticket",
    "it ticket",
    "open a helpdesk",
)


class TicketingError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class TicketingDisabledError(TicketingError):
    def __init__(self) -> None:
        super().__init__("disabled", "Ticketing is disabled")


@dataclass(frozen=True)
class TicketRecord:
    id: str
    tenant_id: UUID
    requester_user_id: UUID
    title: str
    description: str
    priority: str
    status: str
    idempotency_key: str


class TicketAdapter(Protocol):
    def create_ticket(
        self,
        *,
        tenant_id: UUID,
        requester_user_id: UUID,
        title: str,
        description: str,
        priority: str,
        idempotency_key: str,
    ) -> TicketRecord: ...


class MockTicketAdapter:
    """In-process mock. Tickets are not durable across process restarts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: dict[tuple[str, str], TicketRecord] = {}
        self.create_calls = 0

    def create_ticket(
        self,
        *,
        tenant_id: UUID,
        requester_user_id: UUID,
        title: str,
        description: str,
        priority: str,
        idempotency_key: str,
    ) -> TicketRecord:
        if priority not in ALLOWED_PRIORITIES:
            raise TicketingError("invalid_priority", "priority is not allowed")
        cleaned_title = " ".join(title.split())
        cleaned_description = description.strip()
        if len(cleaned_title) < 3 or len(cleaned_title) > 200:
            raise TicketingError("invalid_title", "title must be 3 to 200 characters")
        if not cleaned_description or len(cleaned_description) > 4000:
            raise TicketingError("invalid_description", "description is required")
        key = (str(tenant_id), idempotency_key)
        with self._lock:
            existing = self._by_key.get(key)
            if existing is not None:
                return existing
            self.create_calls += 1
            record = TicketRecord(
                id=str(uuid4()),
                tenant_id=tenant_id,
                requester_user_id=requester_user_id,
                title=cleaned_title,
                description=cleaned_description,
                priority=priority,
                status="open",
                idempotency_key=idempotency_key,
            )
            self._by_key[key] = record
            return record


_adapter: MockTicketAdapter | None = None
_adapter_lock = threading.Lock()


def reset_ticket_adapter() -> None:
    global _adapter
    with _adapter_lock:
        _adapter = None


def get_ticket_adapter() -> TicketAdapter:
    if not settings.ticketing_enabled:
        raise TicketingDisabledError()
    global _adapter
    with _adapter_lock:
        if _adapter is None:
            _adapter = MockTicketAdapter()
        return _adapter


def is_ticket_request(request: str) -> bool:
    lowered = request.lower()
    return any(hint in lowered for hint in TICKET_HINTS)


def ticket_record_payload(record: TicketRecord) -> dict:
    return {
        "simulated": True,
        "created": True,
        "action_type": "create_ticket",
        "ticket_id": record.id,
        "tenant_id": str(record.tenant_id),
        "requester_user_id": str(record.requester_user_id),
        "title": record.title,
        "priority": record.priority,
        "status": record.status,
        "idempotency_key": record.idempotency_key,
        "message": "Created a mock helpdesk ticket. No external vendor API was called.",
    }
