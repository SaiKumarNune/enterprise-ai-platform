"""add tenants, document ownership, ACL, and audit events

Revision ID: b0c1d2e3f4a5
Revises: a9b0c1d2e3f4
Create Date: 2026-10-30 17:40:00.000000

Existing users/documents/workflows are never assigned a tenant unless
LEGACY_TENANT_ID is provided. Verified document owners are applied only from
LEGACY_DOCUMENT_OWNERS / LEGACY_DOCUMENT_OWNERS_BY_FILENAME. Unmapped documents
stay quarantined (owner_id NULL). Blanket LEGACY_DOCUMENT_OWNER_ID is rejected.
"""

from __future__ import annotations

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.services.legacy_backfill import LegacyBackfillError, plan_legacy_backfill

revision: str = "b0c1d2e3f4a5"
down_revision: Union[str, Sequence[str], None] = "a9b0c1d2e3f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("slug", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_tenants_slug"),
    )

    op.add_column("users", sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("documents", sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("documents", sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("agent_workflows", sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=True))

    bind = op.get_bind()
    user_count = bind.execute(sa.text("SELECT count(*) FROM users")).scalar() or 0
    document_rows = bind.execute(sa.text("SELECT id, filename FROM documents")).fetchall()
    user_rows = bind.execute(sa.text("SELECT id FROM users")).fetchall()
    workflow_count = bind.execute(sa.text("SELECT count(*) FROM agent_workflows")).scalar() or 0
    try:
        plan = plan_legacy_backfill(
            user_count=int(user_count),
            document_count=len(document_rows),
            workflow_count=int(workflow_count),
            legacy_tenant_id=os.getenv("LEGACY_TENANT_ID"),
            legacy_tenant_name=os.getenv("LEGACY_TENANT_NAME"),
            legacy_document_owner_id=os.getenv("LEGACY_DOCUMENT_OWNER_ID"),
            default_tenant_id=os.getenv("DEFAULT_TENANT_ID"),
            owner_map_json=os.getenv("LEGACY_DOCUMENT_OWNERS"),
            filename_map_json=os.getenv("LEGACY_DOCUMENT_OWNERS_BY_FILENAME"),
            documents=[(row[0], row[1]) for row in document_rows],
            user_ids=[row[0] for row in user_rows],
        )
    except LegacyBackfillError as exc:
        raise RuntimeError(
            "Refusing to backfill tenant ownership. Existing users, documents, or workflows "
            "require LEGACY_TENANT_ID. Map verified document owners with "
            "LEGACY_DOCUMENT_OWNERS and/or LEGACY_DOCUMENT_OWNERS_BY_FILENAME. "
            "Unmapped documents stay quarantined. Do not set LEGACY_DOCUMENT_OWNER_ID."
        ) from exc

    bind.execute(
        sa.text(
            "INSERT INTO tenants (id, name, slug) VALUES (:id, :name, :slug) "
            "ON CONFLICT (slug) DO NOTHING"
        ),
        {"id": str(plan.tenant_id), "name": plan.tenant_name, "slug": plan.tenant_slug},
    )

    bind.execute(sa.text("UPDATE users SET tenant_id = :tid WHERE tenant_id IS NULL"), {"tid": str(plan.tenant_id)})
    bind.execute(
        sa.text("UPDATE documents SET tenant_id = :tid WHERE tenant_id IS NULL"),
        {"tid": str(plan.tenant_id)},
    )
    for assignment in plan.owner_assignments:
        bind.execute(
            sa.text("UPDATE documents SET owner_id = :oid WHERE id = :did AND owner_id IS NULL"),
            {"oid": str(assignment.owner_id), "did": str(assignment.document_id)},
        )
    bind.execute(
        sa.text(
            "UPDATE agent_workflows AS w SET tenant_id = u.tenant_id "
            "FROM users AS u WHERE w.owner_id = u.id AND w.tenant_id IS NULL"
        )
    )
    bind.execute(
        sa.text("UPDATE agent_workflows SET tenant_id = :tid WHERE tenant_id IS NULL"),
        {"tid": str(plan.tenant_id)},
    )

    remaining_users = bind.execute(sa.text("SELECT count(*) FROM users WHERE tenant_id IS NULL")).scalar()
    remaining_docs = bind.execute(sa.text("SELECT count(*) FROM documents WHERE tenant_id IS NULL")).scalar()
    if remaining_users or remaining_docs:
        raise RuntimeError("Tenant backfill left users or documents without tenant_id; aborting")

    op.alter_column("users", "tenant_id", existing_type=postgresql.UUID(as_uuid=True), nullable=False)
    op.alter_column("documents", "tenant_id", existing_type=postgresql.UUID(as_uuid=True), nullable=False)
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])
    op.create_index("ix_agent_workflows_tenant_id", "agent_workflows", ["tenant_id"])
    op.create_foreign_key("fk_users_tenant_id_tenants", "users", "tenants", ["tenant_id"], ["id"], ondelete="RESTRICT")
    op.create_foreign_key(
        "fk_documents_tenant_id_tenants", "documents", "tenants", ["tenant_id"], ["id"], ondelete="RESTRICT"
    )
    op.create_foreign_key(
        "fk_documents_owner_id_users", "documents", "users", ["owner_id"], ["id"], ondelete="RESTRICT"
    )
    op.create_foreign_key(
        "fk_agent_workflows_tenant_id_tenants",
        "agent_workflows",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_table(
        "document_acl",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("privilege", sa.String(length=20), nullable=False),
        sa.Column("granted_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("privilege IN ('read', 'write', 'manage')", name="ck_document_acl_privilege"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["granted_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "user_id", name="uq_document_acl_document_user"),
    )
    op.create_index("ix_document_acl_document_id", "document_acl", ["document_id"])
    op.create_index("ix_document_acl_user_id", "document_acl", ["user_id"])

    op.create_table(
        "audit_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("actor_tenant_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("resource_type", sa.String(length=40), nullable=True),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("outcome", sa.String(length=20), nullable=False),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_events_actor_user_id", "audit_events", ["actor_user_id"])
    op.create_index("ix_audit_events_actor_tenant_id", "audit_events", ["actor_tenant_id"])
    op.create_index("ix_audit_events_action", "audit_events", ["action"])
    op.create_index("ix_audit_events_resource_id", "audit_events", ["resource_id"])
    op.create_index("ix_audit_events_created_at", "audit_events", ["created_at"])


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("document_acl")
    op.drop_constraint("fk_agent_workflows_tenant_id_tenants", "agent_workflows", type_="foreignkey")
    op.drop_constraint("fk_documents_owner_id_users", "documents", type_="foreignkey")
    op.drop_constraint("fk_documents_tenant_id_tenants", "documents", type_="foreignkey")
    op.drop_constraint("fk_users_tenant_id_tenants", "users", type_="foreignkey")
    op.drop_index("ix_agent_workflows_tenant_id", table_name="agent_workflows")
    op.drop_index("ix_documents_owner_id", table_name="documents")
    op.drop_index("ix_documents_tenant_id", table_name="documents")
    op.drop_index("ix_users_tenant_id", table_name="users")
    op.drop_column("agent_workflows", "tenant_id")
    op.drop_column("documents", "owner_id")
    op.drop_column("documents", "tenant_id")
    op.drop_column("users", "tenant_id")
    op.drop_table("tenants")
