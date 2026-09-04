"""Per-learner, per-concept mastery estimates.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # One row per (learner, concept). `score` is the current belief about how
    # well this learner knows this concept, on 0..1.
    #
    # `attempts` is not a duplicate of COUNT(*) on the attempts table -- it is
    # the evidence counter the learning rate decays on. Keeping it here means
    # the update path reads and writes exactly one row instead of aggregating
    # the whole attempt log on every answer.
    #
    # This table is DERIVED state: it can always be rebuilt by replaying
    # `attempts` through the model. That is why the attempt log stays
    # append-only -- it is the source of truth, this is a cache of a belief.
    op.execute("""
        CREATE TABLE mastery (
            user_id    INT NOT NULL,
            concept_id INT NOT NULL REFERENCES concepts(id),
            score      REAL NOT NULL DEFAULT 0.3 CHECK (score >= 0 AND score <= 1),
            attempts   INT NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (user_id, concept_id)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS mastery")
