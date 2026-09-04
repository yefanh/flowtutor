"""The adaptive engine's database layer: read state, choose, write state.

All of the mathematics lives in `model`. This module only decides which
question to serve and persists the consequences of an answer.
"""

import random
from dataclasses import dataclass

import db
from adaptive import model

EXPLORATION_RATE = 0.20
"""How often to ignore the model and serve something off-target.

Without this the engine is a closed loop that trusts its own estimate. A
learner underrated at the start would be fed easy questions, pass them, gain
almost nothing (an expected win carries no information), and stay underrated
for a long time. Occasionally serving something the model did NOT choose is
what breaks that loop -- the classic explore/exploit tradeoff, and the reason
a purely greedy recommender gets stuck.
"""

REPEAT_PENALTY = 1.5
"""How strongly to avoid re-serving a question the learner has already seen,
measured in difficulty rungs.

A repeat is bad evidence: the learner may be recalling the answer rather than
reasoning. But it is a soft preference, not a rule -- with a small bank, a
perfectly-pitched repeat beats an unseen question three rungs off target.
"""

MAX_REPEAT_PENALTY_STEPS = 2
"""Cap the penalty so a question seen many times is not pushed infinitely far
away; past a couple of sightings the extra evidence of staleness is the same."""

WARMUP_ATTEMPTS = 5
"""Attempts before exploration switches on.

A learner with no history is already being explored -- every concept sits at
the same default and the engine knows nothing. Firing the random branch on top
of that produced a difficulty-5 question as somebody's very first experience,
which is exactly the wrong end of the flow zone to start at.
"""

WEAKNESS_EXPONENT = 2.0
"""Shapes how sharply concept choice favours weak concepts.

Concepts are sampled with weight (1 - mastery) ** WEAKNESS_EXPONENT rather than
always taking the single weakest. The exponent is what makes this behave
differently in the two cases that matter:

  - One weak concept among strong ones -> its weight dwarfs the rest and it is
    chosen almost every time. Effort goes where the gap is.
  - Everything equally weak -> weights are near-equal and concepts interleave.

Always taking the strict minimum failed the second case: a learner who was
struggling everywhere got pinned to whichever concept happened to be a hair
lower, question after question. Interleaving topics also beats blocking them
for retention, so the spread is a feature rather than a concession.
"""


@dataclass(frozen=True)
class ConceptMastery:
    concept_id: int
    concept_name: str
    score: float
    attempts: int
    is_mastered: bool


@dataclass(frozen=True)
class AttemptResult:
    attempt_id: int
    is_correct: bool
    correct_answer: int
    explanation: str | None
    concept_id: int
    concept_name: str
    mastery_before: float
    mastery_after: float
    used_hint: bool
    mastery_updated: bool
    first_attempt: bool
    predicted_probability: float
    crossed_threshold: bool


async def get_mastery(user_id: int) -> list[ConceptMastery]:
    """Every concept with this learner's current estimate.

    Concepts the learner has never touched have no row in `mastery`; they are
    reported at DEFAULT_MASTERY rather than being written on read. Keeping the
    read path free of writes means serving a question never takes a write lock.
    """
    rows = await db.query_all(
        """
        SELECT c.id AS concept_id,
               c.name AS concept_name,
               COALESCE(m.score, %s) AS score,
               COALESCE(m.attempts, 0) AS attempts,
               m.updated_at
        FROM concepts c
        LEFT JOIN mastery m ON m.concept_id = c.id AND m.user_id = %s
        ORDER BY c.id
        """,
        (model.DEFAULT_MASTERY, user_id),
    )
    return [
        ConceptMastery(
            concept_id=r["concept_id"],
            concept_name=r["concept_name"],
            score=float(r["score"]),
            attempts=r["attempts"],
            is_mastered=float(r["score"]) >= model.MASTERY_THRESHOLD,
        )
        for r in rows
    ]


async def _choose_concept(user_id: int) -> tuple[int, float] | None:
    """Choose which concept to practise, and report its current mastery.

    Weighted sampling rather than a strict minimum -- see WEAKNESS_EXPONENT.
    """
    rows = await db.query_all(
        """
        SELECT c.id, COALESCE(m.score, %s) AS score
        FROM concepts c
        LEFT JOIN mastery m ON m.concept_id = c.id AND m.user_id = %s
        ORDER BY c.id
        """,
        (model.DEFAULT_MASTERY, user_id),
    )
    if not rows:
        return None

    scores = [float(r["score"]) for r in rows]
    weights = [(1.0 - s) ** WEAKNESS_EXPONENT for s in scores]

    # Every concept mastered: nothing is weak, so choose uniformly.
    if sum(weights) <= 0:
        weights = [1.0] * len(rows)

    index = random.choices(range(len(rows)), weights=weights, k=1)[0]
    return rows[index]["id"], scores[index]


async def _pick_question(
    user_id: int, concept_id: int | None = None, target: float | None = None
) -> dict | None:
    """Pick one question.

    Targeted mode (concept_id and target given): score every question in the
    concept by how far its difficulty sits from the target, plus a penalty for
    having been seen before, and take the best.

    Exploration mode (neither given): any concept, any difficulty, still
    preferring questions the learner has not seen.

    The `seen` subquery is what keeps the engine from re-serving the same
    question -- a repeat measures recall of the answer, not understanding of
    the concept, so it is contaminated evidence.
    """
    seen_join = """
        LEFT JOIN (
            SELECT question_id, count(*) AS seen
            FROM attempts
            WHERE user_id = %s
            GROUP BY question_id
        ) s ON s.question_id = q.id
    """

    if concept_id is None or target is None:
        where = "TRUE"
        order_by = "COALESCE(s.seen, 0) ASC, random()"
        params: tuple = (user_id,)
    else:
        where = "q.concept_id = %s"
        order_by = "abs(q.difficulty - %s) + %s * LEAST(COALESCE(s.seen, 0), %s) ASC, random()"
        # Order matters: placeholders bind in the order they appear in the
        # final SQL -- the seen-subquery first, then WHERE, then ORDER BY.
        params = (user_id, concept_id, target, REPEAT_PENALTY, MAX_REPEAT_PENALTY_STEPS)

    return await db.query_one(
        f"""
        SELECT q.id, q.concept_id, c.name AS concept_name,
               q.stem, q.options, q.difficulty
        FROM questions q
        JOIN concepts c ON c.id = q.concept_id
        {seen_join}
        WHERE {where}
        ORDER BY {order_by}
        LIMIT 1
        """,
        params,
    )


async def select_question(user_id: int, force_concept_id: int | None = None) -> dict | None:
    """Choose the next question: the heart of the product.

    Two stages, because "what should I practise" and "how hard should it be"
    are different questions:

      1. WHICH CONCEPT -- sampled with weight toward the weak ones, so effort
         goes where the gap is without pinning the learner to one topic.
      2. HOW HARD -- the difficulty at which this learner is predicted to
         succeed TARGET_SUCCESS of the time. Not the hardest they can pass,
         and not something comfortable: the rung where they have to stretch
         but will usually make it.

    Once past warmup, EXPLORATION_RATE of the time both stages are skipped and
    a question is drawn at random, so the engine keeps receiving evidence its
    own estimate did not select for.

    `force_concept_id` pins stage 1 and disables exploration. Teaching mode
    uses it to make the questions right after a lesson be about that lesson.
    """
    if force_concept_id is not None:
        row = await db.query_one(
            """
            SELECT COALESCE(m.score, %s) AS score
            FROM concepts c
            LEFT JOIN mastery m ON m.concept_id = c.id AND m.user_id = %s
            WHERE c.id = %s
            """,
            (model.DEFAULT_MASTERY, user_id, force_concept_id),
        )
        if row is None:
            return None
        question = await _pick_question(
            user_id, force_concept_id, model.target_difficulty(float(row["score"]))
        )
        if question is not None:
            question["exploring"] = False
        return question

    answered = await db.query_one(
        "SELECT count(*) AS n FROM attempts WHERE user_id = %s", (user_id,)
    )
    warmed_up = answered["n"] >= WARMUP_ATTEMPTS
    exploring = warmed_up and random.random() < EXPLORATION_RATE

    if exploring:
        question = await _pick_question(user_id)
    else:
        chosen = await _choose_concept(user_id)
        if chosen is None:
            return None
        concept_id, mastery = chosen
        question = await _pick_question(user_id, concept_id, model.target_difficulty(mastery))

    if question is not None:
        question["exploring"] = exploring
    return question


async def apply_attempt(
    user_id: int,
    question: dict,
    selected: int,
    time_spent: int | None,
    used_hint: bool = False,
) -> AttemptResult:
    """Record an answer and fold it into the mastery estimate.

    Both writes happen in ONE transaction. If they were separate, a crash
    between them would leave an attempt logged but the estimate un-updated --
    and since mastery is derived state, that divergence is silent and
    permanent. The SELECT ... FOR UPDATE closes the other hole: two answers
    submitted at the same moment for the same concept would otherwise both read
    the old score, and one of the two updates would vanish.
    """
    is_correct = selected == question["answer"]
    concept_id = question["concept_id"]

    prior = await db.query_one(
        "SELECT count(*) AS n FROM attempts WHERE user_id = %s AND question_id = %s",
        (user_id, question["id"]),
    )
    first_attempt = prior["n"] == 0

    # A retry that is wrong again is logged but does not move mastery.
    #
    # One question is one piece of evidence about whether the learner knows the
    # concept. Charging for it twice counts the same information twice -- and
    # the retry is a different, easier problem anyway: they now know one option
    # is wrong, so the four-way guess has become a three-way one.
    #
    # A retry that is CORRECT does count, at the reduced hint-assisted rate.
    score_it = first_attempt or is_correct

    async with db.pool.connection() as conn:
        async with conn.transaction():
            cur = await conn.execute(
                """
                INSERT INTO attempts
                    (user_id, question_id, selected, is_correct, time_spent, used_hint)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (user_id, question["id"], selected, is_correct, time_spent, used_hint),
            )
            attempt_id = (await cur.fetchone())["id"]

            # Ensure the row exists so the next statement has something to lock.
            await conn.execute(
                """
                INSERT INTO mastery (user_id, concept_id, score, attempts)
                VALUES (%s, %s, %s, 0)
                ON CONFLICT (user_id, concept_id) DO NOTHING
                """,
                (user_id, concept_id, model.DEFAULT_MASTERY),
            )

            cur = await conn.execute(
                """
                SELECT score, attempts FROM mastery
                WHERE user_id = %s AND concept_id = %s
                FOR UPDATE
                """,
                (user_id, concept_id),
            )
            current = await cur.fetchone()

            result = model.update(
                mastery=float(current["score"]),
                attempts=current["attempts"],
                difficulty=question["difficulty"],
                is_correct=is_correct,
                time_spent=time_spent,
                used_hint=used_hint,
            )

            if score_it:
                await conn.execute(
                    """
                    UPDATE mastery
                    SET score = %s, attempts = attempts + 1, updated_at = now()
                    WHERE user_id = %s AND concept_id = %s
                    """,
                    (result.updated, user_id, concept_id),
                )

    return AttemptResult(
        attempt_id=attempt_id,
        is_correct=is_correct,
        correct_answer=question["answer"],
        explanation=question["explanation"],
        concept_id=concept_id,
        concept_name=question["concept_name"],
        mastery_before=result.previous,
        mastery_after=result.updated if score_it else result.previous,
        used_hint=used_hint,
        mastery_updated=score_it,
        first_attempt=first_attempt,
        predicted_probability=result.predicted_probability,
        crossed_threshold=result.crossed_threshold,
    )
