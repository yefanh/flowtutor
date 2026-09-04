"""Phase 0 API tests.

The one that matters most is test_question_never_leaks_the_answer. Everything
else here is ordinary behaviour; that one guards the security invariant, and it
is the test most likely to catch a real regression when the query in
get_question is edited in a later phase.
"""

import db

TEST_USER = 9001


async def _any_question() -> dict:
    """Read a question straight from the database, answer included.

    Tests are allowed to know the answer -- clients are not. That asymmetry is
    exactly what we are testing.
    """
    return await db.query_one("SELECT id, options, answer FROM questions ORDER BY id LIMIT 1")


async def test_health(client):
    res = await client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "database": "ok"}


async def test_question_never_leaks_the_answer(client):
    res = await client.get("/question", params={"user_id": TEST_USER})
    assert res.status_code == 200
    body = res.json()

    assert "answer" not in body
    assert "explanation" not in body
    assert set(body) == {"id", "concept_id", "concept_name", "stem", "options", "difficulty"}


async def test_correct_answer_is_graded_correct(client):
    question = await _any_question()
    res = await client.post(
        "/answer",
        json={
            "user_id": TEST_USER,
            "question_id": question["id"],
            "selected": question["answer"],
            "time_spent": 12,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["is_correct"] is True
    assert body["correct_answer"] == question["answer"]


async def test_wrong_answer_is_graded_wrong(client):
    question = await _any_question()
    wrong = (question["answer"] + 1) % len(question["options"])
    res = await client.post(
        "/answer",
        json={
            "user_id": TEST_USER,
            "question_id": question["id"],
            "selected": wrong,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["is_correct"] is False
    # The real answer is revealed only after submitting.
    assert body["correct_answer"] == question["answer"]


async def test_attempt_is_persisted(client):
    question = await _any_question()
    res = await client.post(
        "/answer",
        json={
            "user_id": TEST_USER,
            "question_id": question["id"],
            "selected": question["answer"],
            "time_spent": 7,
        },
    )
    attempt_id = res.json()["attempt_id"]

    row = await db.query_one(
        "SELECT user_id, question_id, is_correct, time_spent FROM attempts WHERE id = %s",
        (attempt_id,),
    )
    assert row["user_id"] == TEST_USER
    assert row["question_id"] == question["id"]
    assert row["is_correct"] is True
    assert row["time_spent"] == 7


async def test_out_of_range_option_is_rejected(client):
    question = await _any_question()
    res = await client.post(
        "/answer",
        json={
            "user_id": TEST_USER,
            "question_id": question["id"],
            "selected": 99,
        },
    )
    assert res.status_code == 400


async def test_unknown_question_is_404(client):
    res = await client.post(
        "/answer",
        json={
            "user_id": TEST_USER,
            "question_id": 999999,
            "selected": 0,
        },
    )
    assert res.status_code == 404


async def test_negative_option_is_rejected_by_validation(client):
    res = await client.post(
        "/answer",
        json={
            "user_id": TEST_USER,
            "question_id": 1,
            "selected": -1,
        },
    )
    assert res.status_code == 422
