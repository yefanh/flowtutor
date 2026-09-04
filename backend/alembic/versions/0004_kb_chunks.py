"""Knowledge base chunks for the AI tutor's retrieval.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Retrievable units of source material for the tutor.
    #
    # `key` is a stable identifier derived from where the chunk came from
    # ("lesson:1:4", "explanation:9"). The primary key is a SERIAL that changes
    # every time the knowledge base is rebuilt; the evaluation set has to point
    # at chunks in a way that survives a rebuild, so it references `key`.
    #
    # `embedding` stays NULL until the embedding provider is wired up. 1536
    # matches OpenAI text-embedding-3-small; switching provider means one
    # migration to change the dimension, nothing else.
    #
    # `search_vector` is generated, so keyword search can never drift out of
    # sync with the text it indexes -- there is no application code that could
    # forget to update it.
    op.execute("""
        CREATE TABLE kb_chunks (
            id          SERIAL PRIMARY KEY,
            key         TEXT NOT NULL UNIQUE,
            concept_id  INT NOT NULL REFERENCES concepts(id),
            source_kind TEXT NOT NULL CHECK (source_kind IN ('lesson', 'explanation')),
            source_id   INT NOT NULL,
            title       TEXT,
            content     TEXT NOT NULL,
            embedding   VECTOR(1536),
            search_vector TSVECTOR GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', content), 'B')
            ) STORED
        )
    """)

    # GIN is the index type for full-text search: it maps each lexeme to the
    # rows containing it, which is the shape every text query needs.
    op.execute("""
        CREATE INDEX kb_chunks_search_idx ON kb_chunks USING GIN (search_vector)
    """)
    op.execute("""
        CREATE INDEX kb_chunks_concept_idx ON kb_chunks (concept_id)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS kb_chunks")
