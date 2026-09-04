"""Resize the embedding column to match the chosen model.

Revision ID: 0005
Revises: 0004
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1536 was a placeholder from the spec, sized for OpenAI
    # text-embedding-3-small. The model actually chosen -- BAAI/bge-base-en-v1.5,
    # picked by measurement in evals/embedding_bakeoff.py -- produces 768.
    #
    # Existing vectors are dropped rather than converted. There is nothing to
    # convert: a vector is only meaningful to the model that produced it, and
    # every row is NULL at this point anyway.
    op.execute("DROP INDEX IF EXISTS kb_chunks_embedding_idx")
    op.execute("ALTER TABLE kb_chunks DROP COLUMN embedding")
    op.execute("ALTER TABLE kb_chunks ADD COLUMN embedding VECTOR(768)")

    # HNSW over cosine distance. Deliberately created even though the corpus is
    # tiny: at 29 rows Postgres will ignore it and scan, which is correct, and
    # the index costs nothing until it is worth using.
    #
    # vector_cosine_ops, not L2: BGE vectors are normalised, so cosine is the
    # metric the model was trained for.
    op.execute("""
        CREATE INDEX kb_chunks_embedding_idx
            ON kb_chunks USING hnsw (embedding vector_cosine_ops)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS kb_chunks_embedding_idx")
    op.execute("ALTER TABLE kb_chunks DROP COLUMN embedding")
    op.execute("ALTER TABLE kb_chunks ADD COLUMN embedding VECTOR(1536)")
