"""Hint generation, the answer-leak guardrail, and the reveal flow.

The language model is replaced with a stub throughout. Tests that call a real
API are slow, cost quota, and are not deterministic -- and none of the
behaviour worth pinning down here is the model's. What matters is what we send
it, what we do with what comes back, and what the learner is shown.
"""

import pytest

import db
from adaptive import model
from tutor import guardrail, hints, llm

CACHING = 1


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the model with a scripted one.

    Returns a recorder so tests can assert on what was sent, which is where the
    strongest guarantee lives: the correct answer is never in the prompt.
    """
    sent: list[tuple[str, str]] = []
    replies: list[str] = []

    async def fake_complete(system: str, user: str, max_output_tokens: int = 400):
        sent.append((system, user))
        text = replies.pop(0) if replies else "Have a look at what the lesson says."
        return llm.Completion(
            text=text,
            model="stub",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
        )

    monkeypatch.setattr(llm, "complete", fake_complete)
    monkeypatch.setattr(hints.llm, "complete", fake_complete)
    return {"sent": sent, "replies": replies}


async def _question(difficulty: int = 2) -> dict:
    return await db.query_one(
        """
        SELECT q.id, q.concept_id, c.name AS concept_name,
               q.stem, q.options, q.answer, q.difficulty, q.explanation
        FROM questions q
        JOIN concepts c ON c.id = q.concept_id
        WHERE q.concept_id = %s AND q.difficulty = %s
        LIMIT 1
        """,
        (CACHING, difficulty),
    )


async def _reset(user_id: int) -> None:
    await db.execute("DELETE FROM hints WHERE user_id = %s", (user_id,))
    await db.execute("DELETE FROM attempts WHERE user_id = %s", (user_id,))
    await db.execute("DELETE FROM mastery WHERE user_id = %s", (user_id,))


# ------------------------------------------------------------- the guardrail


def test_guardrail_catches_a_quoted_answer():
    result = guardrail.check(
        "The application code populates it on a cache miss.",
        "The application code, on a cache miss",
        "In the cache-aside pattern, who is responsible for populating the cache?",
    )
    assert result.leaked


def test_guardrail_catches_a_paraphrase_with_a_different_word_form():
    """Caught a real miss: "misses" did not match "miss" before stemming."""
    result = guardrail.check(
        "Remember that the application code is what fills the cache when it misses.",
        "The application code, on a cache miss",
        "In the cache-aside pattern, who is responsible for populating the cache?",
    )
    assert result.leaked


def test_guardrail_allows_a_hint_that_only_points():
    result = guardrail.check(
        "A cache is passive -- it cannot notice anything. See lesson step 6.",
        "The application code, on a cache miss",
        "In the cache-aside pattern, who is responsible for populating the cache?",
    )
    assert not result.leaked


def test_guardrail_ignores_words_the_question_already_contains():
    """Words shared by every option would otherwise flag any on-topic hint."""
    result = guardrail.check(
        "Think about what a cache is for. See lesson step 1.",
        "To reduce read latency and load on the database",
        "What is the primary reason to put a cache in front of a database?",
    )
    assert not result.leaked


# ------------------------------------------------------------- what we send


async def test_the_correct_answer_is_never_in_the_prompt(client, fake_llm):
    """The strongest of the three defences, because it does not depend on
    anything behaving well."""
    user = 9301
    await _reset(user)
    question = await _question()
    wrong = (question["answer"] + 1) % len(question["options"])
    correct_text = question["options"][question["answer"]]

    await hints.generate(user, question, wrong, record=False)

    system, prompt = fake_llm["sent"][0]
    assert correct_text not in prompt
    assert correct_text not in system
    assert question["options"][wrong] in prompt
    # Nor the other distractors -- only the stem and what they chose.
    for i, option in enumerate(question["options"]):
        if i != wrong:
            assert option not in prompt


async def test_a_question_never_grounds_its_own_hint(client, fake_llm):
    user = 9302
    await _reset(user)
    question = await _question()
    wrong = (question["answer"] + 1) % len(question["options"])

    hint = await hints.generate(user, question, wrong, record=False)
    assert f"explanation:{question['id']}" not in hint.sources


# ---------------------------------------------------------- retry and fallback


async def test_a_leaking_hint_is_regenerated(client, fake_llm):
    user = 9303
    await _reset(user)
    question = await _question()
    wrong = (question["answer"] + 1) % len(question["options"])

    fake_llm["replies"].extend(
        [question["options"][question["answer"]], "Take another look at lesson step 6."]
    )
    hint = await hints.generate(user, question, wrong, record=False)

    assert hint.leaked_attempts == 1
    assert not hint.fell_back
    assert "lesson step 6" in hint.text


async def test_persistent_leaking_falls_back_to_pointing(client, fake_llm):
    """Worse teaching than a good hint, but it cannot hand over the answer."""
    user = 9304
    await _reset(user)
    question = await _question()
    wrong = (question["answer"] + 1) % len(question["options"])
    answer_text = question["options"][question["answer"]]

    fake_llm["replies"].extend([answer_text] * hints.MAX_GENERATION_ATTEMPTS)
    hint = await hints.generate(user, question, wrong, record=False)

    assert hint.fell_back
    assert hint.leaked_attempts == hints.MAX_GENERATION_ATTEMPTS
    assert not guardrail.check(hint.text, answer_text, question["stem"]).leaked


# ----------------------------------------------------------------- the flow


async def test_a_first_wrong_answer_does_not_reveal(client, fake_llm):
    """Showing the answer immediately would make the hint pointless."""
    user = 9305
    await _reset(user)
    question = await _question()
    wrong = (question["answer"] + 1) % len(question["options"])

    body = (
        await client.post(
            "/answer",
            json={"user_id": user, "question_id": question["id"], "selected": wrong},
        )
    ).json()

    assert body["is_correct"] is False
    assert body["revealed"] is False
    assert body["correct_answer"] is None
    assert body["explanation"] is None


async def test_a_correct_answer_reveals_immediately(client, fake_llm):
    user = 9306
    await _reset(user)
    question = await _question()

    body = (
        await client.post(
            "/answer",
            json={
                "user_id": user,
                "question_id": question["id"],
                "selected": question["answer"],
            },
        )
    ).json()

    assert body["revealed"] is True
    assert body["correct_answer"] == question["answer"]
    assert body["explanation"]


async def test_the_second_attempt_reveals(client, fake_llm):
    user = 9307
    await _reset(user)
    question = await _question()
    wrong = (question["answer"] + 1) % len(question["options"])
    other = (question["answer"] + 2) % len(question["options"])

    first = (
        await client.post(
            "/answer",
            json={"user_id": user, "question_id": question["id"], "selected": wrong},
        )
    ).json()
    second = (
        await client.post(
            "/answer",
            json={"user_id": user, "question_id": question["id"], "selected": other},
        )
    ).json()

    assert first["revealed"] is False
    assert second["revealed"] is True
    assert second["correct_answer"] == question["answer"]


async def test_hint_endpoint_refuses_a_correct_answer(client, fake_llm):
    question = await _question()
    res = await client.post(
        "/hint",
        json={
            "user_id": 9308,
            "question_id": question["id"],
            "selected": question["answer"],
        },
    )
    assert res.status_code == 400


# ---------------------------------------------------------------- mastery


async def test_a_hinted_correct_answer_earns_less(client, fake_llm):
    user = 9309
    await _reset(user)
    question = await _question()
    wrong = (question["answer"] + 1) % len(question["options"])

    await client.post(
        "/answer",
        json={"user_id": user, "question_id": question["id"], "selected": wrong},
    )
    await client.post(
        "/hint",
        json={"user_id": user, "question_id": question["id"], "selected": wrong},
    )
    hinted = (
        await client.post(
            "/answer",
            json={
                "user_id": user,
                "question_id": question["id"],
                "selected": question["answer"],
            },
        )
    ).json()

    assert hinted["used_hint"] is True
    gain = hinted["mastery"]["delta"]

    unaided = model.update(
        mastery=hinted["mastery"]["previous"],
        attempts=1,
        difficulty=question["difficulty"],
        is_correct=True,
    ).delta
    assert 0 < gain < unaided


async def test_hint_usage_is_read_from_the_server_not_the_client(client, fake_llm):
    """A client cannot claim an assisted answer was unaided."""
    user = 9310
    await _reset(user)
    question = await _question()
    wrong = (question["answer"] + 1) % len(question["options"])

    await client.post(
        "/hint",
        json={"user_id": user, "question_id": question["id"], "selected": wrong},
    )
    assert await hints.was_used(user, question["id"])

    # The request body has no field for it at all, so there is nothing to lie
    # about -- the server looks it up.
    body = (
        await client.post(
            "/answer",
            json={
                "user_id": user,
                "question_id": question["id"],
                "selected": question["answer"],
            },
        )
    ).json()
    assert body["used_hint"] is True


async def test_a_repeated_wrong_answer_is_not_punished_twice(client, fake_llm):
    """One question is one piece of evidence."""
    user = 9311
    await _reset(user)
    question = await _question()
    wrong = (question["answer"] + 1) % len(question["options"])
    other = (question["answer"] + 2) % len(question["options"])

    first = (
        await client.post(
            "/answer",
            json={"user_id": user, "question_id": question["id"], "selected": wrong},
        )
    ).json()
    second = (
        await client.post(
            "/answer",
            json={"user_id": user, "question_id": question["id"], "selected": other},
        )
    ).json()

    assert first["mastery_updated"] is True
    assert first["mastery"]["delta"] < 0

    assert second["mastery_updated"] is False
    assert second["mastery"]["delta"] == 0

    # Both tries are still in the log; only the scoring is skipped.
    logged = await db.query_one(
        "SELECT count(*) AS n FROM attempts WHERE user_id = %s AND question_id = %s",
        (user, question["id"]),
    )
    assert logged["n"] == 2
