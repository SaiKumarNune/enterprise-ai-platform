"""add agent_trace to agent_workflows

Revision ID: f8a2b3c4d5e6
Revises: e7a1b0c2d3f4
Create Date: 2026-10-30 01:25:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f8a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e7a1b0c2d3f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_workflows",
        sa.Column("agent_trace", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_workflows", "agent_trace")
