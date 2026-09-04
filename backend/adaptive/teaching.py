"""Teaching mode: decide when to explain rather than test.

WHY THIS EXISTS
    Mastery alone cannot tell two very different learners apart. One studied
    the concept elsewhere and scores badly. One has never been shown the
    concept at all. Both land at the floor of the scale, and the practice loop
    hands both of them difficulty-1 questions forever.

    Only the first of those is a practice problem. The second is a teaching
    problem, and no amount of well-pitched testing fixes it -- being quizzed on
    something you were never taught teaches almost nothing, and it drives the
    estimate to the floor for the wrong reason.

    So the engine needs a second input besides the score: has this learner
    actually been through the material? That is what lesson_progress answers.

THE THRESHOLD, AND WHY IT IS WHERE IT IS
    Phase 1 measured that five difficulty rungs only resolve mastery between
    roughly 0.37 and 0.87. Below 0.37 the selector clamps to difficulty 1 and
    every question is one the learner is expected to fail -- the mechanism is
    already out of road. That is not a coincidence to work around; it is the
    signal that a different mechanism belongs there. TEACH_BELOW sits at that
    boundary.

WHAT COMPLETING A LESSON DOES NOT DO
    It does not move mastery. Reading an explanation is not evidence of being
    able to do anything, and paying out score for the act of reading would be
    rewarding activity instead of capability -- the exact trap this product is
    built to avoid. Finishing a lesson unlocks practice. Only answers move the
    number.
"""

from dataclasses import dataclass

import db
from adaptive import model

TEACH_BELOW = 0.35
"""Mastery under which a concept is taught rather than tested, provided the
learner has not already been through its lesson."""

PRACTICE_ATTEMPTS_BEFORE_MOVING_ON = 6
"""Escape hatch for the learn-then-use rule below.

A learner who reads a lesson and still cannot answer its questions would
otherwise be held on that one concept forever, never allowed to start anything
new. After this many attempts the engine moves on regardless -- being stuck is
worse than being imperfect, and the concept stays weak so the practice loop
will keep returning to it anyway.
"""


@dataclass(frozen=True)
class LessonStep:
    lesson_id: int
    concept_id: int
    concept_name: str
    step: int
    total_steps: int
    title: str
    body: str


async def concept_awaiting_practice(user_id: int) -> int | None:
    """A concept whose lesson has been read but not yet put to use.

    LEARN ONE THING, USE IT, THEN LEARN THE NEXT.

    Without this rule the engine front-loads all the reading: every concept
    starts below the teaching threshold, so it would walk a learner through
    every lesson of every concept -- thirty-five steps of prose -- before
    asking a single question. Reading five explanations back to back is how
    people bounce off a course.

    It also fixes the smaller version of the same problem seen in the pilot:
    having just finished the caching lesson, the learner was handed a question
    about a concept nobody had taught them. Practising what you just read is
    both the obvious thing to do and the point at which retrieval actually
    strengthens memory.
    """
    row = await db.query_one(
        """
        SELECT c.id
        FROM concepts c
        JOIN lessons l ON l.concept_id = c.id
        LEFT JOIN mastery m ON m.concept_id = c.id AND m.user_id = %s
        WHERE COALESCE(m.score, %s) < %s
          AND COALESCE(m.attempts, 0) < %s
        GROUP BY c.id
        -- Every authored step read, so the lesson is genuinely finished.
        HAVING count(l.id) = (
            SELECT count(*) FROM lesson_progress p
            JOIN lessons l2 ON l2.id = p.lesson_id
            WHERE p.user_id = %s AND l2.concept_id = c.id
        )
        ORDER BY max(
            (SELECT p.completed_at FROM lesson_progress p WHERE p.lesson_id = l.id
              AND p.user_id = %s)
        ) DESC NULLS LAST
        LIMIT 1
        """,
        (
            user_id,
            model.DEFAULT_MASTERY,
            TEACH_BELOW,
            PRACTICE_ATTEMPTS_BEFORE_MOVING_ON,
            user_id,
            user_id,
        ),
    )
    return row["id"] if row else None


async def next_lesson_step(user_id: int) -> LessonStep | None:
    """The next unread lesson step for a concept that needs teaching.

    Returns None when nothing needs teaching, which is the normal state once a
    learner has been through the material -- the caller then falls through to
    the practice loop.

    A concept with no authored lesson steps can never match here, so teaching
    mode switches on per concept as content is written, with no engine change.
    """
    return _to_step(
        await db.query_one(
            """
            WITH needs_teaching AS (
                SELECT c.id, c.name, COALESCE(m.score, %s) AS score
                FROM concepts c
                LEFT JOIN mastery m
                       ON m.concept_id = c.id AND m.user_id = %s
                WHERE COALESCE(m.score, %s) < %s
                  AND EXISTS (SELECT 1 FROM lessons l WHERE l.concept_id = c.id)
            )
            SELECT l.id AS lesson_id,
                   t.id AS concept_id,
                   t.name AS concept_name,
                   l.step,
                   l.title,
                   l.body,
                   (SELECT count(*) FROM lessons
                     WHERE concept_id = t.id) AS total_steps
            FROM needs_teaching t
            JOIN lessons l ON l.concept_id = t.id
            WHERE NOT EXISTS (
                SELECT 1 FROM lesson_progress p
                WHERE p.user_id = %s AND p.lesson_id = l.id
            )
            ORDER BY
                -- Finish the lesson already in progress before opening
                -- another. Without this the ordering below ties on score for
                -- every fresh concept and falls through to `step`, which
                -- interleaves step 1 of everything, then step 2 of
                -- everything -- five lessons read in parallel, none finished.
                EXISTS (
                    SELECT 1 FROM lesson_progress p2
                    JOIN lessons l2 ON l2.id = p2.lesson_id
                    WHERE p2.user_id = %s AND l2.concept_id = t.id
                ) DESC,
                -- Then the weakest concept, with a stable tiebreak so a table
                -- of identical starting scores still settles on one concept
                -- rather than wandering between them.
                t.score ASC,
                t.id ASC,
                -- A lesson is a sequence: step 5 makes no sense before step 4.
                l.step ASC
            LIMIT 1
            """,
            (
                model.DEFAULT_MASTERY,
                user_id,
                model.DEFAULT_MASTERY,
                TEACH_BELOW,
                user_id,
                user_id,
            ),
        )
    )


async def complete_step(user_id: int, lesson_id: int) -> bool:
    """Mark a step as read. Idempotent; returns False if the step is unknown."""
    exists = await db.query_one("SELECT 1 AS ok FROM lessons WHERE id = %s", (lesson_id,))
    if exists is None:
        return False

    await db.execute(
        """
        INSERT INTO lesson_progress (user_id, lesson_id)
        VALUES (%s, %s)
        ON CONFLICT (user_id, lesson_id) DO NOTHING
        """,
        (user_id, lesson_id),
    )
    return True


async def lesson_state(user_id: int) -> list[dict]:
    """Per-concept lesson progress, for the progress panel."""
    return await db.query_all(
        """
        SELECT c.id AS concept_id,
               c.name AS concept_name,
               count(l.id) AS total_steps,
               count(p.lesson_id) AS completed_steps
        FROM concepts c
        LEFT JOIN lessons l ON l.concept_id = c.id
        LEFT JOIN lesson_progress p
               ON p.lesson_id = l.id AND p.user_id = %s
        GROUP BY c.id, c.name
        ORDER BY c.id
        """,
        (user_id,),
    )


def _to_step(row: dict | None) -> LessonStep | None:
    if row is None:
        return None
    return LessonStep(
        lesson_id=row["lesson_id"],
        concept_id=row["concept_id"],
        concept_name=row["concept_name"],
        step=row["step"],
        total_steps=row["total_steps"],
        title=row["title"],
        body=row["body"],
    )
