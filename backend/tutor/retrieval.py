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

from dataclasses import dataclass

import db

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
    return [
        RetrievedChunk(
            key=r["key"],
            concept_id=r["concept_id"],
            source_kind=r["source_kind"],
            source_id=r["source_id"],
            title=r["title"],
            content=r["content"],
            score=float(r["score"]),
        )
        for r in rows
    ]
