"""add users and workflow owner/decision actor

Revision ID: a9b0c1d2e3f4
Revises: f8a2b3c4d5e6
Create Date: 2026-10-30 15:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a9b0c1d2e3f4"
down_revision: Union[str, Sequence[str], None] = "f8a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("role IN ('employee', 'manager', 'admin')", name="ck_users_role"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role", "users", ["role"], unique=False)

    op.add_column(
        "agent_workflows",
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "agent_workflows",
        sa.Column("decided_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index("ix_agent_workflows_owner_id", "agent_workflows", ["owner_id"], unique=False)
    op.create_index(
        "ix_agent_workflows_decided_by_user_id",
        "agent_workflows",
        ["decided_by_user_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_agent_workflows_owner_id_users",
        "agent_workflows",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_agent_workflows_decided_by_user_id_users",
        "agent_workflows",
        "users",
        ["decided_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_agent_workflows_decided_by_user_id_users", "agent_workflows", type_="foreignkey")
    op.drop_constraint("fk_agent_workflows_owner_id_users", "agent_workflows", type_="foreignkey")
    op.drop_index("ix_agent_workflows_decided_by_user_id", table_name="agent_workflows")
    op.drop_index("ix_agent_workflows_owner_id", table_name="agent_workflows")
    op.drop_column("agent_workflows", "decided_by_user_id")
    op.drop_column("agent_workflows", "owner_id")
    op.drop_index("ix_users_role", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
