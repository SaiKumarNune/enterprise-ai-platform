"""Explicit, fail-closed planning for tenant backfill of existing rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID

GREENFIELD_TENANT_ID = UUID("00000000-0000-4000-8000-000000000001")
GREENFIELD_TENANT_SLUG = "default"
GREENFIELD_TENANT_NAME = "Default"


class LegacyBackfillError(ValueError):
    """Existing rows cannot be assigned without an explicit tenant mapping."""


@dataclass(frozen=True)
class OwnerAssignment:
    document_id: UUID
    owner_id: UUID
    source: str


@dataclass(frozen=True)
class BackfillPlan:
    mode: str
    tenant_id: UUID
    tenant_name: str
    tenant_slug: str
    owner_assignments: tuple[OwnerAssignment, ...]
    quarantined_document_ids: tuple[UUID, ...]


def parse_uuid(value: str | None, *, field: str) -> UUID:
    if not value or not str(value).strip():
        raise LegacyBackfillError(f"{field} is required")
    try:
        return UUID(str(value).strip())
    except ValueError as exc:
        raise LegacyBackfillError(f"{field} must be a UUID") from exc


def parse_json_object(raw: str | None, *, field: str) -> dict[str, str]:
    if not raw or not str(raw).strip():
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LegacyBackfillError(f"{field} must be a JSON object") from exc
    if not isinstance(loaded, dict):
        raise LegacyBackfillError(f"{field} must be a JSON object")
    parsed: dict[str, str] = {}
    for key, value in loaded.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise LegacyBackfillError(f"{field} keys and values must be strings")
        parsed[key.strip()] = value.strip()
    return parsed


def plan_legacy_backfill(
    *,
    user_count: int,
    document_count: int,
    workflow_count: int = 0,
    legacy_tenant_id: str | None,
    legacy_tenant_name: str | None,
    default_tenant_id: str | None = None,
    legacy_document_owner_id: str | None = None,
    owner_map_json: str | None = None,
    filename_map_json: str | None = None,
    documents: list[tuple[UUID, str]] | None = None,
    user_ids: list[UUID] | None = None,
) -> BackfillPlan:
    has_legacy = user_count > 0 or document_count > 0 or workflow_count > 0
    if not has_legacy:
        tenant_id = (
            parse_uuid(default_tenant_id, field="DEFAULT_TENANT_ID")
            if default_tenant_id
            else GREENFIELD_TENANT_ID
        )
        return BackfillPlan(
            mode="greenfield",
            tenant_id=tenant_id,
            tenant_name=GREENFIELD_TENANT_NAME,
            tenant_slug=GREENFIELD_TENANT_SLUG,
            owner_assignments=(),
            quarantined_document_ids=(),
        )
    if legacy_document_owner_id and str(legacy_document_owner_id).strip():
        raise LegacyBackfillError(
            "LEGACY_DOCUMENT_OWNER_ID is no longer supported because it assigns every "
            "legacy document to one owner. Use LEGACY_DOCUMENT_OWNERS and/or "
            "LEGACY_DOCUMENT_OWNERS_BY_FILENAME for verified records; unmapped "
            "documents stay quarantined."
        )
    tenant_id = parse_uuid(legacy_tenant_id, field="LEGACY_TENANT_ID")
    name = (legacy_tenant_name or "").strip() or "Legacy"
    rows = documents or []
    known_users = set(user_ids or [])
    if document_count > 0 and len(rows) != document_count:
        raise LegacyBackfillError("document inventory does not match document_count")
    assignments, quarantined = resolve_document_owners(
        documents=rows,
        user_ids=known_users,
        owner_map_json=owner_map_json,
        filename_map_json=filename_map_json,
    )
    return BackfillPlan(
        mode="legacy",
        tenant_id=tenant_id,
        tenant_name=name,
        tenant_slug="legacy",
        owner_assignments=tuple(assignments),
        quarantined_document_ids=tuple(quarantined),
    )


def resolve_document_owners(
    *,
    documents: list[tuple[UUID, str]],
    user_ids: set[UUID],
    owner_map_json: str | None,
    filename_map_json: str | None,
) -> tuple[list[OwnerAssignment], list[UUID]]:
    by_id = parse_json_object(owner_map_json, field="LEGACY_DOCUMENT_OWNERS")
    by_name = parse_json_object(filename_map_json, field="LEGACY_DOCUMENT_OWNERS_BY_FILENAME")
    known_ids = {item[0] for item in documents}
    filenames: dict[str, list[UUID]] = {}
    for document_id, filename in documents:
        filenames.setdefault(filename, []).append(document_id)

    assignments: dict[UUID, OwnerAssignment] = {}

    for raw_doc_id, raw_owner_id in by_id.items():
        document_id = parse_uuid(raw_doc_id, field="LEGACY_DOCUMENT_OWNERS document id")
        owner_id = parse_uuid(raw_owner_id, field="LEGACY_DOCUMENT_OWNERS owner id")
        if document_id not in known_ids:
            raise LegacyBackfillError(f"LEGACY_DOCUMENT_OWNERS references unknown document {document_id}")
        if owner_id not in user_ids:
            raise LegacyBackfillError(f"LEGACY_DOCUMENT_OWNERS owner {owner_id} is not an existing user")
        assignments[document_id] = OwnerAssignment(document_id, owner_id, "id")

    for filename, raw_owner_id in by_name.items():
        matches = filenames.get(filename, [])
        if not matches:
            raise LegacyBackfillError(
                f"LEGACY_DOCUMENT_OWNERS_BY_FILENAME references unknown filename {filename!r}"
            )
        if len(matches) > 1:
            raise LegacyBackfillError(
                f"filename {filename!r} is not unique; map those documents by UUID instead"
            )
        owner_id = parse_uuid(raw_owner_id, field="LEGACY_DOCUMENT_OWNERS_BY_FILENAME owner id")
        if owner_id not in user_ids:
            raise LegacyBackfillError(
                f"LEGACY_DOCUMENT_OWNERS_BY_FILENAME owner {owner_id} is not an existing user"
            )
        document_id = matches[0]
        if document_id not in assignments:
            assignments[document_id] = OwnerAssignment(document_id, owner_id, "filename")

    quarantined = [document_id for document_id, _name in documents if document_id not in assignments]
    return list(assignments.values()), quarantined


def rollback_legacy_document_state(
    documents: list[dict],
) -> list[dict]:
    """In-memory inverse of the document tenant/owner backfill for tests."""
    reverted = []
    for item in documents:
        reverted.append({**item, "tenant_id": None, "owner_id": None})
    return reverted
