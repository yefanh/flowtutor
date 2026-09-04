"""Long-term memory the tutor writes about a learner, and agent traces.

Revision ID: 0007
Revises: 0006
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # What the tutor has noticed about this learner that is not derivable from
    # the data it already has.
    #
    # Mastery scores and past answers are already in the database, so nothing
    # needs to be copied here. What cannot be computed is the interpretation:
    # "confuses caching with persistence", "keeps reaching for capacity when
    # the problem is invalidation". The tutor writes those itself, through a
    # tool, when it notices a pattern.
    op.execute("""
        CREATE TABLE learner_memory (
            user_id    INT NOT NULL,
            key        TEXT NOT NULL,
            value      JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (user_id, key)
        )
    """)

    # Every step the agent took, in order: which tool it chose, with what
    # arguments, and what came back.
    #
    # An agent is several decisions deep, and when the output is wrong the
    # question is always WHICH step went wrong -- a bad search, a bad reading
    # of good material, or a bad final write-up. Without the trace all three
    # look identical from the outside.
    op.execute("""
        ALTER TABLE hints ADD COLUMN trace JSONB NOT NULL DEFAULT '[]'::jsonb
    """)
    op.execute("""
        ALTER TABLE hints ADD COLUMN steps INT NOT NULL DEFAULT 0
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE hints DROP COLUMN steps")
    op.execute("ALTER TABLE hints DROP COLUMN trace")
    op.execute("DROP TABLE IF EXISTS learner_memory")
