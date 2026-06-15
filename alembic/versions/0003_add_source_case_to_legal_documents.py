"""add source case fields to legal_documents

Revision ID: 0003_source_case_id
Revises: 0002_add_legal_documents
Create Date: 2026-06-14
"""

from alembic import op
import sqlalchemy as sa

revision = "0003_source_case_id"
down_revision = "0002_add_legal_documents"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE legal_documents "
        "ADD COLUMN IF NOT EXISTS source_case_id varchar(80) NOT NULL DEFAULT ''"
    )
    op.execute(
        "ALTER TABLE legal_documents "
        "ADD COLUMN IF NOT EXISTS source_chunk_index integer NOT NULL DEFAULT 0"
    )
    op.execute(
        "UPDATE legal_documents SET source_case_id = id WHERE source_case_id = ''"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_legal_docs_source_case_id "
        "ON legal_documents (source_case_id)"
    )


def downgrade():
    op.drop_index("idx_legal_docs_source_case_id", table_name="legal_documents")
    op.drop_column("legal_documents", "source_chunk_index")
    op.drop_column("legal_documents", "source_case_id")
