"""add pgvector extension and chunk embeddings

Revision ID: c3a91f7e2b10
Revises: 9d572e3d04d5
Create Date: 2026-10-29 20:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "c3a91f7e2b10"
down_revision: Union[str, Sequence[str], None] = "9d572e3d04d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    op.add_column(
        "document_chunks",
        sa.Column("embedding", Vector(384), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "embedding")
    # The vector extension is left installed; other objects may depend on it.
