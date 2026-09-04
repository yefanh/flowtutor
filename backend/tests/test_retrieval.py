"""Knowledge base and retrieval tests.

The last test is the important one: it pins the measured retrieval quality so a
change that quietly makes search worse fails here instead of showing up later
as vague hints.
"""

import pytest

import db
from evals import retrieval_eval
from tutor import knowledge_base, retrieval

CACHING = 1

# Measured baseline for keyword-only search: Recall@5 0.806, Hit@5 0.875.
# The floors sit below that so ordinary noise does not fail the build, but a
# real regression does. Raise them when hybrid retrieval lands.
MIN_RECALL_AT_5 = 0.75
MIN_HIT_RATE_AT_5 = 0.83


@pytest.fixture(scope="session", autouse=True)
async def _knowledge_base(_pool):
    """Build the corpus once before any retrieval test reads from it."""
    await knowledge_base.rebuild()


# --------------------------------------------------------------- the corpus


async def test_rebuild_is_idempotent():
    first = await knowledge_base.rebuild()
    second = await knowledge_base.rebuild()
    assert first["total"] == second["total"]
    assert second["removed"] == 0


async def test_every_lesson_step_and_explanation_becomes_a_chunk():
    chunks = await knowledge_base.collect_chunks()
    lessons = await db.query_one("SELECT count(*) AS n FROM lessons")
    explanations = await db.query_one(
        "SELECT count(*) AS n FROM questions WHERE explanation IS NOT NULL"
    )
    assert len(chunks) == lessons["n"] + explanations["n"]


async def test_changing_the_text_clears_the_stored_embedding():
    """A vector computed from text that no longer exists still matches queries,
    just for content nobody will ever read."""
    key = "lesson:1:1"
    fake = "[" + ",".join(["0.1"] * 1536) + "]"
    await db.execute("UPDATE kb_chunks SET embedding = %s WHERE key = %s", (fake, key))

    original = await db.query_one("SELECT body FROM lessons WHERE concept_id=1 AND step=1")
    try:
        await db.execute(
            "UPDATE lessons SET body = %s WHERE concept_id=1 AND step=1",
            (original["body"] + " Revised.",),
        )
        await knowledge_base.rebuild()
        row = await db.query_one("SELECT embedding FROM kb_chunks WHERE key = %s", (key,))
        assert row["embedding"] is None
    finally:
        await db.execute(
            "UPDATE lessons SET body = %s WHERE concept_id=1 AND step=1",
            (original["body"],),
        )
        await knowledge_base.rebuild()


# -------------------------------------------------------------- keyword search


async def test_finds_the_right_lesson_step():
    hits = await retrieval.keyword_search("thundering herd stampede", limit=5)
    assert "lesson:1:7" in [c.key for c in hits]


async def test_natural_language_queries_return_results():
    """OR semantics. Postgres helpers AND every term together, which returns
    nothing at all for a sentence -- the failure this guards against."""
    hits = await retrieval.keyword_search(
        "why does my cache still show the old value after an update"
    )
    assert hits


async def test_a_question_never_retrieves_its_own_explanation():
    """That explanation is a restatement of the answer. Handing it to the
    tutor is handing over the answer to give away."""
    question = await db.query_one(
        "SELECT id, stem FROM questions WHERE concept_id = %s ORDER BY id LIMIT 1",
        (CACHING,),
    )
    own_key = f"explanation:{question['id']}"

    unfiltered = await retrieval.keyword_search(question["stem"], limit=10)
    assert own_key in [c.key for c in unfiltered], "expected it to rank without the filter"

    filtered = await retrieval.keyword_search(
        question["stem"], limit=10, exclude_question_id=question["id"]
    )
    assert own_key not in [c.key for c in filtered]


async def test_concept_filter_restricts_results():
    hits = await retrieval.keyword_search("consistency", limit=10, concept_id=CACHING)
    assert all(c.concept_id == CACHING for c in hits)


async def test_empty_query_returns_nothing_rather_than_erroring():
    assert await retrieval.keyword_search("   ") == []
    assert await retrieval.keyword_search("the and of") == []


async def test_lesson_chunks_cite_their_step():
    hits = await retrieval.keyword_search("thundering herd", limit=5)
    lesson = next(c for c in hits if c.source_kind == "lesson")
    assert "lesson step" in lesson.citation


# ------------------------------------------------------------------ the ruler


async def test_golden_set_points_at_chunks_that_exist():
    """A typo'd key looks exactly like a retrieval failure in the scores."""
    assert await retrieval_eval.validate_golden_set() == []


async def test_retrieval_quality_has_not_regressed():
    result = await retrieval_eval.evaluate("keyword", retrieval_eval.METHODS["keyword"], k=5)
    assert result.recall >= MIN_RECALL_AT_5, f"Recall@5 fell to {result.recall:.3f}"
    assert result.hit_rate >= MIN_HIT_RATE_AT_5, f"Hit@5 fell to {result.hit_rate:.3f}"
