"""add agent_workflows table

Revision ID: e7a1b0c2d3f4
Revises: c3a91f7e2b10
Create Date: 2026-10-30 00:50:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e7a1b0c2d3f4"
down_revision: Union[str, Sequence[str], None] = "c3a91f7e2b10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_workflows",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("request", sa.Text(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=True),
        sa.Column("top_k", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("intent", sa.String(length=40), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("draft", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("retrieval_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("grounded", sa.Boolean(), nullable=True),
        sa.Column("citation_valid", sa.Boolean(), nullable=True),
        sa.Column("abstained", sa.Boolean(), nullable=True),
        sa.Column("approval_decision", sa.String(length=20), nullable=True),
        sa.Column("execution_result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("graph_state", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_workflows_status", "agent_workflows", ["status"])


def downgrade() -> None:
    op.drop_index("ix_agent_workflows_status", table_name="agent_workflows")
    op.drop_table("agent_workflows")
