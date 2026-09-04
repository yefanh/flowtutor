"""Lesson content and per-learner lesson progress.

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # An ordered sequence of teaching steps for a concept.
    #
    # This exists because mastery alone could not distinguish two very
    # different learners: one who studied the concept and scores badly, and one
    # who has never been shown the concept at all. Both sat at the floor, and
    # both got handed difficulty-1 questions forever. Only the first of those
    # is a practice problem; the second needs to be taught.
    #
    # Authoring lesson steps for a concept is what switches teaching mode on
    # for it -- a concept with no rows here goes straight to practice, so
    # content can be added one concept at a time without touching the engine.
    op.execute("""
        CREATE TABLE lessons (
            id         SERIAL PRIMARY KEY,
            concept_id INT NOT NULL REFERENCES concepts(id),
            step       INT NOT NULL CHECK (step >= 1),
            title      TEXT NOT NULL,
            body       TEXT NOT NULL,
            UNIQUE (concept_id, step)
        )
    """)

    # Which steps a learner has worked through.
    #
    # Deliberately NOT a mastery signal. Reading an explanation is not evidence
    # of being able to do anything, so completing a lesson moves nobody's
    # score -- it only unlocks practice. Rewarding the act of reading would be
    # rewarding activity instead of capability, which is the trap this whole
    # product is built to avoid.
    op.execute("""
        CREATE TABLE lesson_progress (
            user_id      INT NOT NULL,
            lesson_id    INT NOT NULL REFERENCES lessons(id),
            completed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (user_id, lesson_id)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS lesson_progress")
    op.execute("DROP TABLE IF EXISTS lessons")
