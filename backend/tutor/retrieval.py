"""Finding the material a hint should be grounded in.

Retrieval quality is the ceiling on answer quality. A tutor cannot explain
using material it never found, and it will happily invent something to fill
the gap -- so this module is measured, not eyeballed. See
`evals/run_retrieval_eval.py`.

Right now there is one method here: keyword search. Dense (embedding) search
and the fusion of the two arrive once an embedding provider is configured. The
evaluation harness exists first so those additions can be judged against a
number rather than a hunch.
"""

import asyncio
from dataclasses import dataclass

import db
from tutor import embedding, reranking

DEFAULT_LIMIT = 5


@dataclass(frozen=True)
class RetrievedChunk:
    key: str
    concept_id: int
    source_kind: str
    source_id: int
    title: str | None
    content: str
    score: float

    @property
    def citation(self) -> str:
        """How a hint should refer to this material out loud."""
        if self.source_kind == "lesson":
            _, _, step = self.key.split(":")
            return f"lesson step {step}: {self.title}"
        return "a related question"


async def keyword_search(
    query: str,
    limit: int = DEFAULT_LIMIT,
    concept_id: int | None = None,
    exclude_question_id: int | None = None,
) -> list[RetrievedChunk]:
    """Full-text search over the corpus.

    OR SEMANTICS, NOT AND. This is the thing that makes or breaks full-text
    search on natural-language input. Postgres helpers such as
    `websearch_to_tsquery` and `plainto_tsquery` join every term with AND:
    "why does my cache still show the old value after an update" becomes
    `cach & still & show & old & valu & updat`, and no chunk contains all six,
    so a perfectly reasonable question returns nothing at all. Measured
    directly: every test query returned zero rows before this was changed.

    Real ranked retrieval -- BM25 included -- does the opposite. Match ANY
    term, then rank by how many and how strongly. So the query text is run
    through `to_tsvector` to get normalised lexemes and those are rejoined with
    `|`. Going through the tsvector also means the terms are already lexemes,
    so nothing user-typed is interpolated into query syntax.

    `ts_rank_cd` weighs term frequency and how close the matched terms sit to
    each other. It is NOT BM25: Postgres full-text ranking has no document
    length normalisation, so it under-penalises long documents. Good enough
    while chunks are of similar size, which they are here. If measurement later
    shows this is the weak link, real BM25 needs the pg_search extension.

    `exclude_question_id` keeps a learner's current question out of its own
    hint. That question's explanation is a restatement of its answer.
    """
    # An empty or all-stopword query produces an empty tsquery, which
    # to_tsquery rejects. Nothing to search for is not an error.
    if not query.strip():
        return []

    filters = ["search_vector @@ q"]
    params: list = [query]

    if concept_id is not None:
        filters.append("concept_id = %s")
        params.append(concept_id)

    if exclude_question_id is not None:
        filters.append("NOT (source_kind = 'explanation' AND source_id = %s)")
        params.append(exclude_question_id)

    params.append(limit)

    rows = await db.query_all(
        f"""
        SELECT key, concept_id, source_kind, source_id, title, content,
               ts_rank_cd(search_vector, q) AS score
        FROM kb_chunks,
             to_tsquery(
                 'english',
                 array_to_string(
                     tsvector_to_array(to_tsvector('english', %s)), ' | '
                 )
             ) AS q
        WHERE {" AND ".join(filters)}
        ORDER BY score DESC, key ASC
        LIMIT %s
        """,
        tuple(params),
    )
    return [_to_chunk(r) for r in rows]


FUSION_DEPTH = 20
"""How deep each method searches before its results are fused.

Fusion can only promote a chunk that at least one method surfaced, so a chunk
sitting at rank 14 in keyword search and rank 3 in dense search is rescuable
only if both lists run that deep. Costs nothing but a slightly larger SELECT.
"""

RERANK_POOL = 10
"""How many fused candidates the reranker actually reads.

Separate from FUSION_DEPTH, and the two must not be collapsed into one number.
Searching deeper is nearly free; reranking deeper is not -- the cross-encoder
is the slowest step and its cost is linear in this value. Measured, holding
fusion depth at 20:

    reranked   Recall@5   Hit@5     MRR   rerank ms
           5      0.847   0.875   0.819          86
          10      0.924   1.000   0.892         131
          20      0.924   1.000   0.889         260
          30      0.924   1.000   0.889         269

Quality plateaus at 10 while latency keeps climbing, so anything above it is
paid for and unused. Below it the pool is too thin to hold the right chunk.

This distinction was found by breaking it: collapsing both into a single
constant of 10 shrank the search depth too, and Recall@5 fell from 0.924 to
0.882 with no other change.
"""

RRF_K = 60
"""The damping constant in reciprocal rank fusion. 60 is the value from the
original paper and is not sensitive -- it mostly controls how sharply rank 1
outweighs rank 10."""


async def dense_search(
    query: str,
    limit: int = DEFAULT_LIMIT,
    concept_id: int | None = None,
    exclude_question_id: int | None = None,
) -> list[RetrievedChunk]:
    """Vector search: match on meaning rather than on shared words.

    This is what answers the queries keyword search could not. "how long should
    something stay in the cache" shares no word with an answer that talks about
    TTL, time to live and lifetimes -- lexically they are strangers, and
    semantically they are the same question.

    `<=>` is pgvector's cosine distance. Score is reported as 1 - distance so
    that, like every other method here, larger is better.
    """
    vector = str(await embedding.embed_query(query))

    filters = ["embedding IS NOT NULL"]
    params: list = [vector]

    if concept_id is not None:
        filters.append("concept_id = %s")
        params.append(concept_id)
    if exclude_question_id is not None:
        filters.append("NOT (source_kind = 'explanation' AND source_id = %s)")
        params.append(exclude_question_id)

    params.extend([vector, limit])

    rows = await db.query_all(
        f"""
        SELECT key, concept_id, source_kind, source_id, title, content,
               1 - (embedding <=> %s::vector) AS score
        FROM kb_chunks
        WHERE {" AND ".join(filters)}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        tuple(params),
    )
    return [_to_chunk(r) for r in rows]


async def hybrid_search(
    query: str,
    limit: int = DEFAULT_LIMIT,
    concept_id: int | None = None,
    exclude_question_id: int | None = None,
) -> list[RetrievedChunk]:
    """Keyword and dense search, fused.

    The two fail in different ways, which is the entire argument for running
    both. Dense search understands that "how long should something stay in the
    cache" is asking about TTL; keyword search nails an exact rare term like
    "cache-aside" that a vector may blur into its neighbours.

    FUSED BY RANK, NOT BY SCORE. `ts_rank_cd` and cosine similarity are
    numbers on unrelated scales -- one is an unbounded relevance weight, the
    other is bounded roughly 0..1 -- so adding or averaging them is arithmetic
    on incomparable units, and whichever happens to have the larger range wins
    by accident. Reciprocal rank fusion throws the scores away and uses only
    position:

        score(chunk) = sum over methods of 1 / (RRF_K + rank in that method)

    A chunk both methods rank highly beats one that either ranks first alone,
    and no normalisation, tuning or calibration is needed to make that true.
    """
    keyword, dense = await asyncio.gather(
        keyword_search(query, FUSION_DEPTH, concept_id, exclude_question_id),
        dense_search(query, FUSION_DEPTH, concept_id, exclude_question_id),
    )

    fused: dict[str, float] = {}
    seen: dict[str, RetrievedChunk] = {}
    for results in (keyword, dense):
        for rank, chunk in enumerate(results, start=1):
            fused[chunk.key] = fused.get(chunk.key, 0.0) + 1.0 / (RRF_K + rank)
            seen.setdefault(chunk.key, chunk)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [RetrievedChunk(**{**vars(seen[key]), "score": score}) for key, score in ordered[:limit]]


async def rerank(
    query: str, candidates: list[RetrievedChunk], limit: int = DEFAULT_LIMIT
) -> list[RetrievedChunk]:
    """Re-score candidates by reading each one against the query.

    Retrieval and reranking do different jobs. Retrieval compares a query
    vector to document vectors that were computed without ever seeing the
    query -- fast enough to run over a whole corpus, and necessarily
    approximate, because the document had to be summarised into a vector before
    anyone knew what would be asked. A cross-encoder reads the query and the
    document *together* and scores that pair directly. Far more accurate, far
    too slow to run over everything, which is exactly why it goes last and only
    over a shortlist.
    """
    if not candidates:
        return []

    scores = await reranking.score(query, [c.content for c in candidates])
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: -p[1])
    return [
        RetrievedChunk(**{**vars(chunk), "score": float(score)}) for chunk, score in ranked[:limit]
    ]


async def search(
    query: str,
    limit: int = DEFAULT_LIMIT,
    concept_id: int | None = None,
    exclude_question_id: int | None = None,
) -> list[RetrievedChunk]:
    """The retrieval the tutor actually calls: hybrid, then reranked."""
    candidates = await hybrid_search(query, RERANK_POOL, concept_id, exclude_question_id)
    return await rerank(query, candidates, limit)


def _to_chunk(row: dict) -> RetrievedChunk:
    return RetrievedChunk(
        key=row["key"],
        concept_id=row["concept_id"],
        source_kind=row["source_kind"],
        source_id=row["source_id"],
        title=row["title"],
        content=row["content"],
        score=float(row["score"]),
    )
