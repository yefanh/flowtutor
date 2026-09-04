"""Tests for teaching mode.

The behaviour being pinned down: a learner who has never seen a concept gets
taught it, not quizzed on it -- and reading the lesson unlocks practice without
awarding any mastery.
"""

import db
from adaptive import model, teaching

TAUGHT_CONCEPT = 1  # Caching, the only concept with authored lesson content


async def _reset(user_id: int) -> None:
    await db.execute("DELETE FROM attempts WHERE user_id = %s", (user_id,))
    await db.execute("DELETE FROM mastery WHERE user_id = %s", (user_id,))
    await db.execute("DELETE FROM lesson_progress WHERE user_id = %s", (user_id,))


async def _read_whole_lesson(user_id: int, concept_id: int) -> int:
    steps = await db.query_all(
        "SELECT id FROM lessons WHERE concept_id = %s ORDER BY step", (concept_id,)
    )
    for step in steps:
        await teaching.complete_step(user_id, step["id"])
    return len(steps)


async def test_a_new_learner_is_taught_not_tested(client):
    user = 9201
    await _reset(user)

    res = await client.get("/next", params={"user_id": user})
    assert res.status_code == 200
    body = res.json()

    assert body["kind"] == "lesson"
    assert body["question"] is None
    assert body["lesson"]["step"] == 1
    assert body["lesson"]["concept_name"] == "Caching"


async def test_lesson_steps_arrive_in_order(client):
    user = 9202
    await _reset(user)

    seen = []
    for _ in range(20):
        body = (await client.get("/next", params={"user_id": user})).json()
        if body["kind"] != "lesson":
            break
        seen.append(body["lesson"]["step"])
        await client.post(
            "/lesson/complete",
            json={"user_id": user, "lesson_id": body["lesson"]["lesson_id"]},
        )

    total = (
        seen[0]
        and (
            await db.query_one(
                "SELECT count(*) AS n FROM lessons WHERE concept_id = %s", (TAUGHT_CONCEPT,)
            )
        )["n"]
    )
    assert seen == list(range(1, total + 1))


async def test_practice_starts_once_the_lesson_is_read(client):
    user = 9203
    await _reset(user)
    await _read_whole_lesson(user, TAUGHT_CONCEPT)

    body = (await client.get("/next", params={"user_id": user})).json()
    assert body["kind"] == "question"
    assert body["lesson"] is None


async def test_reading_a_lesson_awards_no_mastery(client):
    """Reading is not evidence of capability. Only answers move the number."""
    user = 9204
    await _reset(user)

    before = (await client.get("/mastery", params={"user_id": user})).json()
    await _read_whole_lesson(user, TAUGHT_CONCEPT)
    after = (await client.get("/mastery", params={"user_id": user})).json()

    scores_before = {c["concept_id"]: c["score"] for c in before}
    scores_after = {c["concept_id"]: c["score"] for c in after}
    assert scores_before == scores_after
    assert all(c["attempts"] == 0 for c in after)


async def test_completing_a_step_is_idempotent(client):
    user = 9205
    await _reset(user)

    step = await db.query_one(
        "SELECT id FROM lessons WHERE concept_id = %s ORDER BY step LIMIT 1",
        (TAUGHT_CONCEPT,),
    )
    for _ in range(3):
        res = await client.post("/lesson/complete", json={"user_id": user, "lesson_id": step["id"]})
        assert res.status_code == 200

    rows = await db.query_one(
        "SELECT count(*) AS n FROM lesson_progress WHERE user_id = %s", (user,)
    )
    assert rows["n"] == 1


async def test_unknown_lesson_step_is_404(client):
    res = await client.post("/lesson/complete", json={"user_id": 9206, "lesson_id": 999999})
    assert res.status_code == 404


async def test_a_strong_learner_is_never_sent_back_to_the_lesson(client):
    """Teaching is for learners below the threshold. Someone who already scores
    well should go straight to practice even having read nothing."""
    user = 9207
    await _reset(user)

    await db.execute(
        """
        INSERT INTO mastery (user_id, concept_id, score, attempts)
        VALUES (%s, %s, 0.75, 10)
        ON CONFLICT (user_id, concept_id) DO UPDATE SET score = EXCLUDED.score
        """,
        (user, TAUGHT_CONCEPT),
    )
    # Every other concept too, so nothing else can claim the teaching slot.
    others = await db.query_all("SELECT id FROM concepts WHERE id <> %s", (TAUGHT_CONCEPT,))
    for concept in others:
        await db.execute(
            """
            INSERT INTO mastery (user_id, concept_id, score, attempts)
            VALUES (%s, %s, 0.75, 10)
            ON CONFLICT (user_id, concept_id) DO UPDATE SET score = EXCLUDED.score
            """,
            (user, concept["id"]),
        )

    assert await teaching.next_lesson_step(user) is None
    body = (await client.get("/next", params={"user_id": user})).json()
    assert body["kind"] == "question"


async def test_concepts_without_lesson_content_are_never_taught(client):
    """Authoring lesson steps is what switches teaching on for a concept, so a
    concept with no content must fall through to practice rather than block."""
    user = 9208
    await _reset(user)
    await _read_whole_lesson(user, TAUGHT_CONCEPT)

    untaught = await db.query_all(
        "SELECT id FROM concepts WHERE id NOT IN (SELECT concept_id FROM lessons)"
    )
    assert untaught, "test assumes at least one concept has no lesson yet"

    # All of them are at the default, well under the teaching threshold.
    assert model.DEFAULT_MASTERY < teaching.TEACH_BELOW
    assert await teaching.next_lesson_step(user) is None


async def test_mastery_endpoint_reports_lesson_progress(client):
    user = 9209
    await _reset(user)

    step = await db.query_one(
        "SELECT id FROM lessons WHERE concept_id = %s ORDER BY step LIMIT 1",
        (TAUGHT_CONCEPT,),
    )
    await teaching.complete_step(user, step["id"])

    body = (await client.get("/mastery", params={"user_id": user})).json()
    caching = next(c for c in body if c["concept_id"] == TAUGHT_CONCEPT)

    assert caching["lesson_steps_total"] == 7
    assert caching["lesson_steps_done"] == 1

    others = [c for c in body if c["concept_id"] != TAUGHT_CONCEPT]
    assert all(c["lesson_steps_total"] == 0 for c in others)


# ------------------------------------------------- learn one thing, use it


async def _add_lesson(concept_id: int, steps: int = 2) -> None:
    """Give a second concept lesson content, so ordering across concepts can be
    tested without waiting for that content to be authored for real."""
    for step in range(1, steps + 1):
        await db.execute(
            """
            INSERT INTO lessons (concept_id, step, title, body)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (concept_id, step) DO NOTHING
            """,
            (concept_id, step, f"Test step {step}", "Body."),
        )


async def _drop_lessons(concept_id: int) -> None:
    await db.execute(
        """
        DELETE FROM lesson_progress
        WHERE lesson_id IN (SELECT id FROM lessons WHERE concept_id = %s)
        """,
        (concept_id,),
    )
    await db.execute("DELETE FROM lessons WHERE concept_id = %s", (concept_id,))


async def test_practice_follows_the_lesson_just_read(client):
    """Having read about caching, the next question should be about caching --
    not about a concept nobody has taught yet."""
    user = 9210
    await _reset(user)
    await _read_whole_lesson(user, TAUGHT_CONCEPT)

    body = (await client.get("/next", params={"user_id": user})).json()
    assert body["kind"] == "question"
    assert body["question"]["concept_id"] == TAUGHT_CONCEPT


async def test_lessons_are_not_all_front_loaded(client):
    """The wall-of-reading failure.

    Every concept starts below the teaching threshold, so a naive rule reads
    every lesson of every concept before asking anything. The order must
    instead be: teach A, practise A, teach B.
    """
    user = 9211
    second = 2  # Message Queues
    await _reset(user)
    await _drop_lessons(second)
    await _add_lesson(second, steps=2)

    try:
        order = []
        for _ in range(14):
            body = (await client.get("/next", params={"user_id": user})).json()
            if body["kind"] == "lesson":
                order.append(("lesson", body["lesson"]["concept_id"]))
                await client.post(
                    "/lesson/complete",
                    json={"user_id": user, "lesson_id": body["lesson"]["lesson_id"]},
                )
            else:
                question = body["question"]
                order.append(("question", question["concept_id"]))
                answer = await db.query_one(
                    "SELECT answer FROM questions WHERE id = %s", (question["id"],)
                )
                await client.post(
                    "/answer",
                    json={
                        "user_id": user,
                        "question_id": question["id"],
                        "selected": answer["answer"],
                        "time_spent": 10,
                    },
                )

        kinds = [kind for kind, _ in order]
        first_question = kinds.index("question")
        lessons_before_first_question = order[:first_question]

        # Only ONE concept may be taught before practice begins.
        taught_first = {concept for _, concept in lessons_before_first_question}
        assert len(taught_first) == 1

        # And the first question is about that same concept.
        assert order[first_question][1] == taught_first.pop()

        # The second concept's lesson must appear only after practice started.
        second_lessons = [
            i for i, (kind, c) in enumerate(order) if kind == "lesson" and c == second
        ]
        assert second_lessons, "second lesson never appeared"
        assert min(second_lessons) > first_question
    finally:
        await _drop_lessons(second)


async def test_a_stuck_learner_is_eventually_released(client):
    """A learner who reads the lesson and still cannot answer must not be held
    on that one concept forever."""
    user = 9212
    await _reset(user)
    await _read_whole_lesson(user, TAUGHT_CONCEPT)

    # Wrong answers, so mastery never clears the teaching threshold.
    for _ in range(teaching.PRACTICE_ATTEMPTS_BEFORE_MOVING_ON):
        body = (await client.get("/next", params={"user_id": user})).json()
        assert body["kind"] == "question"
        question = body["question"]
        answer = await db.query_one(
            "SELECT answer, options FROM questions WHERE id = %s", (question["id"],)
        )
        await client.post(
            "/answer",
            json={
                "user_id": user,
                "question_id": question["id"],
                "selected": (answer["answer"] + 1) % len(answer["options"]),
            },
        )

    stored = await db.query_one(
        "SELECT score FROM mastery WHERE user_id = %s AND concept_id = %s",
        (user, TAUGHT_CONCEPT),
    )
    tried = await db.query_one(
        """
        SELECT count(*) AS n FROM attempts a
        JOIN questions q ON q.id = a.question_id
        WHERE a.user_id = %s AND q.concept_id = %s
        """,
        (user, TAUGHT_CONCEPT),
    )
    assert float(stored["score"]) < teaching.TEACH_BELOW  # still stuck
    assert tried["n"] >= teaching.PRACTICE_ATTEMPTS_BEFORE_MOVING_ON

    # ...but no longer pinned to it.
    assert await teaching.concept_awaiting_practice(user) is None
