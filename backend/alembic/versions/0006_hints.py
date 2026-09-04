"""Hint records, and marking attempts that followed one.

Revision ID: 0006
Revises: 0005
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Every hint served, with what grounded it.
    #
    # Two jobs. First, observability: `sources` records which chunks the hint
    # was built from, so a bad hint can be traced back to bad retrieval rather
    # than guessed at. `leaked_attempts` counts how often the guardrail caught
    # the model stating the answer and made it try again -- a number that
    # should stay near zero and is worth watching if it does not.
    #
    # Second, and the reason this is a table rather than a log line: whether a
    # learner used a hint decides how much mastery a subsequent correct answer
    # earns. That has to be recorded server-side. A client that reported its own
    # hint usage could claim full credit for an assisted answer.
    op.execute("""
        CREATE TABLE hints (
            id              SERIAL PRIMARY KEY,
            user_id         INT NOT NULL,
            question_id     INT NOT NULL REFERENCES questions(id),
            selected        INT,
            hint            TEXT NOT NULL,
            sources         TEXT[] NOT NULL DEFAULT '{}',
            model           TEXT,
            leaked_attempts INT NOT NULL DEFAULT 0,
            latency_ms      INT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX hints_user_question_idx ON hints (user_id, question_id)
    """)

    # Denormalised onto the attempt so the mastery update has it without a join,
    # and so the historical record says what was true at the time.
    op.execute("""
        ALTER TABLE attempts
        ADD COLUMN used_hint BOOLEAN NOT NULL DEFAULT FALSE
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE attempts DROP COLUMN used_hint")
    op.execute("DROP TABLE IF EXISTS hints")
