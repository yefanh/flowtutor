"""Knowledge base and retrieval tests.

The last test is the important one: it pins the measured retrieval quality so a
change that quietly makes search worse fails here instead of showing up later
as vague hints.
"""

import pytest

import db
from evals import retrieval_eval
from tutor import embedding, knowledge_base, retrieval

CACHING = 1

# Measured on the golden set, hybrid + rerank: Recall@5 0.924, Hit@5 1.000,
# MRR 0.892. The floors sit just below so a real regression fails the build.
# Both models are deterministic, so these numbers do not drift on their own.
MIN_RECALL_AT_5 = 0.88
MIN_HIT_RATE_AT_5 = 0.95


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


async def test_changing_the_text_reembeds_the_chunk():
    """A vector computed from text that no longer exists still matches queries,
    just for content nobody will ever read.

    The mechanism is that the upsert nulls the embedding whenever the text it
    was derived from changed, and the backfill then recomputes it. What is
    observable, and what this checks, is that the stored vector is no longer
    the one that was there before the edit.
    """
    key = "lesson:1:1"
    # Read the dimension rather than hardcoding it: this test broke once
    # already when the chosen model changed and 1536 became 768.
    planted = "[" + ",".join(["0.1"] * embedding.DIMENSIONS) + "]"
    await db.execute("UPDATE kb_chunks SET embedding = %s WHERE key = %s", (planted, key))

    original = await db.query_one("SELECT body FROM lessons WHERE concept_id=1 AND step=1")
    try:
        await db.execute(
            "UPDATE lessons SET body = %s WHERE concept_id=1 AND step=1",
            (original["body"] + " Revised.",),
        )
        await knowledge_base.rebuild()

        row = await db.query_one("SELECT embedding FROM kb_chunks WHERE key = %s", (key,))
        assert row["embedding"] is not None
        assert str(row["embedding"]) != planted
    finally:
        await db.execute(
            "UPDATE lessons SET body = %s WHERE concept_id=1 AND step=1",
            (original["body"],),
        )
        await knowledge_base.rebuild()


async def test_unchanged_text_keeps_its_embedding():
    """Embeddings are not free to recompute, so a rebuild that changes nothing
    must not re-embed the whole corpus."""
    before = await db.query_one("SELECT embedding FROM kb_chunks WHERE key = 'lesson:1:2'")
    stats = await knowledge_base.rebuild()
    after = await db.query_one("SELECT embedding FROM kb_chunks WHERE key = 'lesson:1:2'")

    assert stats["embedded"] == 0
    assert str(before["embedding"]) == str(after["embedding"])


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
    """The number the whole tutor rests on. Hint quality cannot exceed the
    quality of the material retrieval puts in front of the model."""
    result = await retrieval_eval.evaluate(
        "hybrid + rerank", retrieval_eval.METHODS["hybrid + rerank"], k=5
    )
    assert result.recall >= MIN_RECALL_AT_5, f"Recall@5 fell to {result.recall:.3f}"
    assert result.hit_rate >= MIN_HIT_RATE_AT_5, f"Hit@5 fell to {result.hit_rate:.3f}"


async def test_every_query_finds_something_relevant():
    """Hit@5 of 1.000 means no query produces a hint with nothing behind it.
    A single miss here is a learner getting an invented answer."""
    result = await retrieval_eval.evaluate(
        "hybrid + rerank", retrieval_eval.METHODS["hybrid + rerank"], k=5
    )
    assert result.misses == []


# ----------------------------------------------------------- dense and hybrid


async def test_dense_search_finds_meaning_without_shared_words():
    """The case keyword search cannot do.

    The query says "how long should something stay"; the answer says TTL, time
    to live, lifetime, sixty seconds. Not one word in common.
    """
    query = "how long should something stay in the cache"

    keyword_keys = [c.key for c in await retrieval.keyword_search(query)]
    dense_keys = [c.key for c in await retrieval.dense_search(query)]

    assert "lesson:1:5" not in keyword_keys, "expected keyword search to miss this"
    assert "lesson:1:5" in dense_keys


async def test_every_chunk_has_a_vector():
    row = await db.query_one("SELECT count(*) AS n FROM kb_chunks WHERE embedding IS NULL")
    assert row["n"] == 0


async def test_hybrid_returns_results_either_method_found():
    query = "what happens when a popular key expires"
    keyword = {c.key for c in await retrieval.keyword_search(query, limit=10)}
    dense = {c.key for c in await retrieval.dense_search(query, limit=10)}
    hybrid = {c.key for c in await retrieval.hybrid_search(query, limit=10)}
    assert hybrid <= (keyword | dense)
    assert hybrid


async def test_reranking_reorders_the_candidates():
    query = "what is a thundering herd"
    candidates = await retrieval.hybrid_search(query, limit=retrieval.RERANK_POOL)
    reranked = await retrieval.rerank(query, candidates, limit=5)

    assert [c.key for c in reranked] != [c.key for c in candidates[:5]]
    assert reranked[0].key in {"lesson:1:7", "explanation:3"}


async def test_dense_search_also_excludes_a_question_from_its_own_hint():
    """The leak has to be closed on every retrieval path, not just the first
    one that was written."""
    question = await db.query_one(
        "SELECT id, stem FROM questions WHERE concept_id = %s ORDER BY id LIMIT 1",
        (CACHING,),
    )
    own_key = f"explanation:{question['id']}"

    unfiltered = await retrieval.dense_search(question["stem"], limit=10)
    assert own_key in [c.key for c in unfiltered]

    for method in (retrieval.dense_search, retrieval.hybrid_search, retrieval.search):
        results = await method(question["stem"], limit=10, exclude_question_id=question["id"])
        assert own_key not in [c.key for c in results], method.__name__
