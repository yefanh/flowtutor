"""Phase 1 integration tests: the engine against a real database.

The model is tested in isolation elsewhere. What is checked here is the wiring
-- that answers actually move stored mastery, that selection reads it back,
and that the two writes in an answer stay consistent.

Each test uses its own user id so the shared development database does not
make them interfere.
"""

import pytest

import db
from adaptive import engine, model


async def _reset(user_id: int) -> None:
    await db.execute("DELETE FROM attempts WHERE user_id = %s", (user_id,))
    await db.execute("DELETE FROM mastery WHERE user_id = %s", (user_id,))


async def _question_of_difficulty(difficulty: int) -> dict:
    return await db.query_one(
        """
        SELECT q.id, q.concept_id, c.name AS concept_name,
               q.options, q.answer, q.difficulty, q.explanation
        FROM questions q
        JOIN concepts c ON c.id = q.concept_id
        WHERE q.difficulty = %s
        ORDER BY q.id
        LIMIT 1
        """,
        (difficulty,),
    )


async def test_new_learner_starts_at_the_default(client):
    user = 9101
    await _reset(user)

    res = await client.get("/mastery", params={"user_id": user})
    assert res.status_code == 200
    body = res.json()

    assert len(body) == 5  # every seeded concept is reported
    assert all(c["score"] == model.DEFAULT_MASTERY for c in body)
    assert all(c["attempts"] == 0 for c in body)
    assert not any(c["is_mastered"] for c in body)


async def test_correct_answer_raises_stored_mastery(client):
    user = 9102
    await _reset(user)
    question = await _question_of_difficulty(3)

    res = await client.post(
        "/answer",
        json={
            "user_id": user,
            "question_id": question["id"],
            "selected": question["answer"],
        },
    )
    delta = res.json()["mastery"]

    assert delta["current"] > delta["previous"]
    assert delta["delta"] > 0

    stored = await db.query_one(
        "SELECT score, attempts FROM mastery WHERE user_id = %s AND concept_id = %s",
        (user, question["concept_id"]),
    )
    # `score` is REAL (4-byte float, ~7 significant digits) while Python uses
    # doubles, so a round trip through Postgres loses the low bits. At ~1e-8
    # per update on a 0..1 value that is far below anything the model cares
    # about, so the column stays REAL and the test compares approximately.
    assert float(stored["score"]) == pytest.approx(delta["current"], abs=1e-6)
    assert stored["attempts"] == 1


async def test_wrong_answer_lowers_stored_mastery(client):
    user = 9103
    await _reset(user)
    question = await _question_of_difficulty(1)
    wrong = (question["answer"] + 1) % len(question["options"])

    res = await client.post(
        "/answer",
        json={"user_id": user, "question_id": question["id"], "selected": wrong},
    )
    delta = res.json()["mastery"]

    assert delta["current"] < delta["previous"]


async def test_repeated_success_climbs_and_unlocks_harder_questions(client):
    """The end-to-end promise of the phase: get things right, get harder work.

    Answers are applied through the engine directly rather than through the
    selector, so the test controls the difficulty and is not at the mercy of
    the exploration coin flip.
    """
    user = 9104
    await _reset(user)
    question = await _question_of_difficulty(2)
    concept_id = question["concept_id"]

    start = model.target_difficulty(model.DEFAULT_MASTERY)

    for _ in range(12):
        await engine.apply_attempt(user, question, question["answer"], time_spent=8)

    stored = await db.query_one(
        "SELECT score FROM mastery WHERE user_id = %s AND concept_id = %s",
        (user, concept_id),
    )
    end = model.target_difficulty(float(stored["score"]))

    assert float(stored["score"]) > model.DEFAULT_MASTERY
    assert end > start


async def test_selection_concentrates_on_a_lone_weak_concept(client):
    """Effort should go where the gap is -- when there is a clear gap."""
    user = 9105
    await _reset(user)

    concepts = await db.query_all("SELECT id FROM concepts ORDER BY id")
    weakest = concepts[-1]["id"]

    # Everyone is strong except the last concept.
    for c in concepts:
        score = 0.2 if c["id"] == weakest else 0.9
        await db.execute(
            """
            INSERT INTO mastery (user_id, concept_id, score, attempts)
            VALUES (%s, %s, %s, 5)
            ON CONFLICT (user_id, concept_id) DO UPDATE SET score = EXCLUDED.score
            """,
            (user, c["id"], score),
        )

    # Weighted sampling, so this is a strong tendency rather than a guarantee.
    # With one concept at 0.2 against four at 0.9 the weights are 0.64 vs 0.01
    # each, and exploration only fires 20% of the time -- a clear majority is
    # expected and the margin is wide enough not to be flaky.
    picks = [await engine.select_question(user) for _ in range(60)]
    on_target = sum(1 for p in picks if p["concept_id"] == weakest)
    assert on_target > 36


async def test_selection_never_locks_onto_one_concept(client):
    """Even with a clear weakest concept, the engine must keep sampling others.

    Two mechanisms produce this -- weighted sampling and the exploration
    branch. What matters for correctness is the outcome: a closed loop that
    only ever revisits its own choice stops receiving evidence that could
    correct a bad estimate.
    """
    user = 9106
    await _reset(user)

    concepts = await db.query_all("SELECT id FROM concepts ORDER BY id")
    weakest = concepts[-1]["id"]
    for c in concepts:
        await db.execute(
            """
            INSERT INTO mastery (user_id, concept_id, score, attempts)
            VALUES (%s, %s, %s, 5)
            ON CONFLICT (user_id, concept_id) DO UPDATE SET score = EXCLUDED.score
            """,
            (user, c["id"], 0.2 if c["id"] == weakest else 0.9),
        )

    picks = [await engine.select_question(user) for _ in range(120)]
    off_target = sum(1 for p in picks if p["concept_id"] != weakest)
    assert off_target > 0


async def test_selection_avoids_questions_already_seen(client):
    """A repeat measures recall of the answer, not understanding."""
    user = 9107
    await _reset(user)

    concept = await db.query_one(
        "SELECT concept_id FROM questions GROUP BY concept_id ORDER BY count(*) DESC LIMIT 1"
    )
    concept_id = concept["concept_id"]
    questions = await db.query_all(
        "SELECT id, difficulty FROM questions WHERE concept_id = %s ORDER BY difficulty",
        (concept_id,),
    )
    target = float(questions[0]["difficulty"])

    first = await engine._pick_question(user, concept_id, target)

    # Log an attempt on it, then ask again with the same target.
    await db.execute(
        """
        INSERT INTO attempts (user_id, question_id, selected, is_correct)
        VALUES (%s, %s, 0, TRUE)
        """,
        (user, first["id"]),
    )
    second = await engine._pick_question(user, concept_id, target)

    assert second["id"] != first["id"]


async def test_attempt_and_mastery_are_written_together(client):
    """Both writes are in one transaction, so counts must never diverge."""
    user = 9108
    await _reset(user)
    question = await _question_of_difficulty(4)

    for _ in range(5):
        await engine.apply_attempt(user, question, question["answer"], time_spent=10)

    attempts = await db.query_one("SELECT count(*) AS n FROM attempts WHERE user_id = %s", (user,))
    mastery = await db.query_one(
        "SELECT attempts AS n FROM mastery WHERE user_id = %s AND concept_id = %s",
        (user, question["concept_id"]),
    )
    assert attempts["n"] == mastery["n"] == 5


async def test_a_learner_weak_everywhere_gets_variety(client):
    """The failure this weighting exists to prevent.

    Strict weakest-first pinned a struggling learner to whichever concept sat
    a hair below the others, question after question. When nothing stands out,
    concepts should interleave instead.
    """
    user = 9109
    await _reset(user)

    concepts = await db.query_all("SELECT id FROM concepts ORDER BY id")
    for i, c in enumerate(concepts):
        await db.execute(
            """
            INSERT INTO mastery (user_id, concept_id, score, attempts)
            VALUES (%s, %s, %s, 10)
            ON CONFLICT (user_id, concept_id) DO UPDATE SET score = EXCLUDED.score
            """,
            # Barely distinguishable, which is what broke the old rule.
            (user, c["id"], 0.05 + i * 0.01),
        )

    picks = [await engine.select_question(user) for _ in range(60)]
    distinct = {p["concept_id"] for p in picks}
    assert len(distinct) == len(concepts)


async def test_no_exploration_before_warmup(client):
    """A learner with no history should never be handed a random hard question
    as their first experience."""
    user = 9110
    await _reset(user)

    picks = [await engine.select_question(user) for _ in range(30)]
    assert all(p["exploring"] is False for p in picks)


async def test_exploration_switches_on_after_warmup(client):
    user = 9111
    await _reset(user)

    question = await db.query_one("SELECT id FROM questions LIMIT 1")
    for _ in range(engine.WARMUP_ATTEMPTS):
        await db.execute(
            """
            INSERT INTO attempts (user_id, question_id, selected, is_correct)
            VALUES (%s, %s, 0, TRUE)
            """,
            (user, question["id"]),
        )

    picks = [await engine.select_question(user) for _ in range(120)]
    assert any(p["exploring"] for p in picks)
