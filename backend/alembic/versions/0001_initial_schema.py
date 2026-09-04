"""Initial schema: concepts, questions, attempts.

Phase 0 tables only. Later phases add their own migrations:
    Phase 1: mastery
    Phase 2: kb_chunks (pgvector)
    Phase 3: learner_memory

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enabled now so Phase 2 does not need a separate superuser step later.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # A concept is one learnable unit of the domain (Caching, Message Queues...).
    # Mastery in Phase 1 is tracked PER CONCEPT, which is why a question must
    # belong to exactly one concept: the concept is the unit the adaptive engine
    # reasons about. A question spanning two concepts would leave the engine
    # unable to decide whose mastery score to move.
    op.execute("""
        CREATE TABLE concepts (
            id          SERIAL PRIMARY KEY,
            name        TEXT NOT NULL UNIQUE,
            description TEXT
        )
    """)

    # One practice item.
    #   options: JSON array of choice strings
    #   answer:  0-based index into options. NEVER sent to the client.
    #   difficulty: 1..5, the column the Phase 1 adaptive engine selects on.
    #
    # UNIQUE (concept_id, stem) is what makes seed.sql re-runnable: reloading
    # content is an ON CONFLICT DO NOTHING away instead of a duplicate bank.
    op.execute("""
        CREATE TABLE questions (
            id          SERIAL PRIMARY KEY,
            concept_id  INT NOT NULL REFERENCES concepts(id),
            stem        TEXT NOT NULL,
            options     JSONB NOT NULL,
            answer      INT NOT NULL,
            difficulty  INT NOT NULL DEFAULT 1 CHECK (difficulty BETWEEN 1 AND 5),
            explanation TEXT,
            UNIQUE (concept_id, stem)
        )
    """)

    # Phase 1 queries this as "questions in concept X near difficulty Y".
    op.execute("""
        CREATE INDEX questions_concept_difficulty_idx
            ON questions (concept_id, difficulty)
    """)

    # Append-only log of every answer submitted. This is the raw signal that
    # both the adaptive engine (Phase 1) and the agent's long-term memory
    # (Phase 3) are derived from, so rows here are never updated or deleted --
    # rewriting history would make every derived estimate unreproducible.
    op.execute("""
        CREATE TABLE attempts (
            id          SERIAL PRIMARY KEY,
            user_id     INT NOT NULL,
            question_id INT NOT NULL REFERENCES questions(id),
            selected    INT NOT NULL,
            is_correct  BOOLEAN NOT NULL,
            time_spent  INT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE INDEX attempts_user_created_idx
            ON attempts (user_id, created_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS attempts")
    op.execute("DROP TABLE IF EXISTS questions")
    op.execute("DROP TABLE IF EXISTS concepts")
