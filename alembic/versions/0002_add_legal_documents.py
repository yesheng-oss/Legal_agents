"""add legal_documents table with pgvector

Revision ID: 0002_add_legal_documents
Revises: 0001_initial_postgresql_schema
Create Date: 2026-06-10
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0002_add_legal_documents"
down_revision = "0001_initial_postgresql_schema"
branch_labels = None
depends_on = None


def upgrade():
    # 启用扩展
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "legal_documents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("accusations", sa.Text(), nullable=False, server_default=""),
        sa.Column("articles", sa.Text(), nullable=False, server_default=""),
        sa.Column("punishment", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_case_id", sa.String(length=80), nullable=False, server_default=""),
        sa.Column("source_chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding", Vector(512), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # 向量相似度索引（ivfflat，适合中等规模数据）
    op.execute(
        "CREATE INDEX idx_legal_docs_embedding ON legal_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    # pg_trgm 文本相似度索引
    op.execute(
        "CREATE INDEX idx_legal_docs_content_trgm ON legal_documents USING gin (content gin_trgm_ops)"
    )
    # 罪名过滤索引
    op.execute(
        "CREATE INDEX idx_legal_docs_accusations ON legal_documents USING gin (accusations gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX idx_legal_docs_source_case_id ON legal_documents (source_case_id)"
    )


def downgrade():
    op.drop_table("legal_documents")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
    op.execute("DROP EXTENSION IF EXISTS vector")
