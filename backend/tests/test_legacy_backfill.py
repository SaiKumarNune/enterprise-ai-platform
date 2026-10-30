from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.services.legacy_backfill import (
    GREENFIELD_TENANT_ID,
    LegacyBackfillError,
    plan_legacy_backfill,
    rollback_legacy_document_state,
)

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "alembic/versions/b0c1d2e3f4a5_add_tenants_acl_and_audit.py"
)


def test_greenfield_backfill_uses_documented_default_tenant() -> None:
    plan = plan_legacy_backfill(
        user_count=0,
        document_count=0,
        workflow_count=0,
        legacy_tenant_id=None,
        legacy_tenant_name=None,
    )
    assert plan.mode == "greenfield"
    assert plan.tenant_id == GREENFIELD_TENANT_ID
    assert plan.owner_assignments == ()
    assert plan.quarantined_document_ids == ()


def test_legacy_users_without_tenant_id_fail_closed() -> None:
    with pytest.raises(LegacyBackfillError, match="LEGACY_TENANT_ID"):
        plan_legacy_backfill(
            user_count=2,
            document_count=0,
            workflow_count=0,
            legacy_tenant_id=None,
            legacy_tenant_name=None,
        )


def test_blanket_legacy_owner_is_rejected() -> None:
    with pytest.raises(LegacyBackfillError, match="LEGACY_DOCUMENT_OWNER_ID is no longer supported"):
        plan_legacy_backfill(
            user_count=2,
            document_count=3,
            workflow_count=17,
            legacy_tenant_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            legacy_tenant_name="Legacy Co",
            legacy_document_owner_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            documents=[],
            user_ids=[],
        )


def test_unmapped_documents_are_quarantined_and_verified_owners_are_assigned() -> None:
    handbook = uuid4()
    resume = uuid4()
    vacation = uuid4()
    owner = uuid4()
    plan = plan_legacy_backfill(
        user_count=2,
        document_count=3,
        workflow_count=17,
        legacy_tenant_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        legacy_tenant_name="Legacy Co",
        filename_map_json='{"vacation_policy.txt":"%s","Sai_Kumar_Nune_AI.pdf":"%s"}' % (owner, owner),
        documents=[
            (handbook, "employee_handbook.pdf"),
            (resume, "Sai_Kumar_Nune_AI.pdf"),
            (vacation, "vacation_policy.txt"),
        ],
        user_ids=[owner, uuid4()],
    )
    assigned = {item.document_id: item.owner_id for item in plan.owner_assignments}
    assert assigned == {resume: owner, vacation: owner}
    assert plan.quarantined_document_ids == (handbook,)


def test_unknown_mapping_key_fails_closed() -> None:
    owner = uuid4()
    with pytest.raises(LegacyBackfillError, match="unknown document"):
        plan_legacy_backfill(
            user_count=1,
            document_count=1,
            legacy_tenant_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            legacy_tenant_name="Legacy",
            owner_map_json='{"%s":"%s"}' % (uuid4(), owner),
            documents=[(uuid4(), "vacation_policy.txt")],
            user_ids=[owner],
        )


def test_unknown_filename_mapping_fails_closed() -> None:
    owner = uuid4()
    doc_id = uuid4()
    with pytest.raises(LegacyBackfillError, match="unknown filename"):
        plan_legacy_backfill(
            user_count=1,
            document_count=1,
            legacy_tenant_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            legacy_tenant_name="Legacy",
            filename_map_json='{"missing.txt":"%s"}' % owner,
            documents=[(doc_id, "vacation_policy.txt")],
            user_ids=[owner],
        )


def test_owner_must_be_an_existing_user() -> None:
    doc_id = uuid4()
    with pytest.raises(LegacyBackfillError, match="not an existing user"):
        plan_legacy_backfill(
            user_count=1,
            document_count=1,
            legacy_tenant_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            legacy_tenant_name="Legacy",
            owner_map_json='{"%s":"%s"}' % (doc_id, uuid4()),
            documents=[(doc_id, "vacation_policy.txt")],
            user_ids=[uuid4()],
        )


def test_duplicate_filename_mapping_fails_closed() -> None:
    owner = uuid4()
    with pytest.raises(LegacyBackfillError, match="not unique"):
        plan_legacy_backfill(
            user_count=1,
            document_count=2,
            legacy_tenant_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            legacy_tenant_name="Legacy",
            filename_map_json='{"dup.txt":"%s"}' % owner,
            documents=[(uuid4(), "dup.txt"), (uuid4(), "dup.txt")],
            user_ids=[owner],
        )


def test_invalid_json_mapping_fails_closed() -> None:
    with pytest.raises(LegacyBackfillError, match="JSON object"):
        plan_legacy_backfill(
            user_count=1,
            document_count=0,
            legacy_tenant_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            legacy_tenant_name="Legacy",
            owner_map_json="[1, 2]",
            user_ids=[uuid4()],
        )


def test_in_memory_rollback_clears_tenant_and_owner() -> None:
    original = [{"id": "1", "tenant_id": "t", "owner_id": "o", "filename": "a.pdf"}]
    reverted = rollback_legacy_document_state(original)
    assert reverted == [{"id": "1", "tenant_id": None, "owner_id": None, "filename": "a.pdf"}]
    assert original[0]["owner_id"] == "o"


def test_migration_keeps_owner_nullable_and_downgrade_drops_new_tables() -> None:
    source = MIGRATION.read_text(encoding="utf-8")
    upgrade, downgrade = source.split("def downgrade", 1)
    assert 'op.alter_column("documents", "owner_id"' not in upgrade
    assert "nullable=True" in upgrade
    assert 'op.drop_column("documents", "owner_id")' in downgrade
    assert 'op.drop_table("tenants")' in downgrade
    assert 'op.drop_table("document_acl")' in downgrade
    assert 'op.drop_table("audit_events")' in downgrade
