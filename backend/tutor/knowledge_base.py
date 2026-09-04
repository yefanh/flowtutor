"""Building the retrievable corpus.

WHERE THE MATERIAL COMES FROM
    Nothing new was written for this. The corpus is assembled from material
    that already had to exist:

      * lesson steps    -- the teaching content
      * question explanations -- why each answer is right

    That is not a shortcut, it is the point. The lesson steps were authored one
    idea per step, which is exactly the unit a retrieval chunk wants to be, so
    chunking is a no-op here. Splitting long prose into overlapping windows --
    normally the fiddliest part of building a RAG corpus, and the part that
    quietly caps retrieval quality -- simply does not arise.

THE LEAK
    A question's explanation is, by construction, a restatement of its answer.
    Retrieve the explanation for the very question the learner is stuck on and
    the tutor is handed the answer to give away.

    So every retrieval call can exclude a specific question's own material.
    A prompt instruction not to reveal the answer is a request; not putting the
    answer in front of the model is a guarantee. Both are used.
"""

from dataclasses import dataclass

import db
from tutor import embedding


@dataclass(frozen=True)
class Chunk:
    key: str
    concept_id: int
    source_kind: str
    source_id: int
    title: str | None
    content: str


async def collect_chunks() -> list[Chunk]:
    """Read the source material and turn it into chunks.

    Pure assembly -- no writes, so it can be inspected on its own.
    """
    chunks: list[Chunk] = []

    lessons = await db.query_all(
        "SELECT id, concept_id, step, title, body FROM lessons ORDER BY concept_id, step"
    )
    for lesson in lessons:
        chunks.append(
            Chunk(
                key=f"lesson:{lesson['concept_id']}:{lesson['step']}",
                concept_id=lesson["concept_id"],
                source_kind="lesson",
                source_id=lesson["id"],
                title=lesson["title"],
                content=lesson["body"],
            )
        )

    explanations = await db.query_all(
        """
        SELECT id, concept_id, stem, explanation
        FROM questions
        WHERE explanation IS NOT NULL
        ORDER BY id
        """
    )
    for question in explanations:
        chunks.append(
            Chunk(
                key=f"explanation:{question['id']}",
                concept_id=question["concept_id"],
                source_kind="explanation",
                # The question id, which is what lets retrieval exclude a
                # learner's current question from its own hint.
                source_id=question["id"],
                title=question["stem"],
                content=question["explanation"],
            )
        )

    return chunks


async def rebuild() -> dict[str, int]:
    """Sync the knowledge base with the current source material.

    Upserts by `key` rather than wiping and reinserting, because embeddings
    cost money: a chunk whose text has not changed keeps the vector already
    computed for it.

    When the text HAS changed, the stored embedding is set back to NULL. A
    vector computed from text that no longer exists is worse than no vector --
    it still matches queries, just for content nobody will ever read.
    """
    chunks = await collect_chunks()

    for chunk in chunks:
        await db.execute(
            """
            INSERT INTO kb_chunks
                (key, concept_id, source_kind, source_id, title, content)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (key) DO UPDATE SET
                concept_id  = EXCLUDED.concept_id,
                source_kind = EXCLUDED.source_kind,
                source_id   = EXCLUDED.source_id,
                title       = EXCLUDED.title,
                content     = EXCLUDED.content,
                embedding   = CASE
                    WHEN kb_chunks.content IS DISTINCT FROM EXCLUDED.content
                      OR kb_chunks.title   IS DISTINCT FROM EXCLUDED.title
                    THEN NULL
                    ELSE kb_chunks.embedding
                END
            """,
            (
                chunk.key,
                chunk.concept_id,
                chunk.source_kind,
                chunk.source_id,
                chunk.title,
                chunk.content,
            ),
        )

    # Drop chunks whose source material was deleted, so the corpus cannot
    # accumulate orphans that still turn up in search results.
    keys = [chunk.key for chunk in chunks]
    removed = await db.query_one(
        """
        WITH deleted AS (
            DELETE FROM kb_chunks WHERE key <> ALL(%s) RETURNING 1
        )
        SELECT count(*) AS n FROM deleted
        """,
        (keys,),
    )

    embedded = await backfill_embeddings()

    stats = await db.query_one(
        """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE embedding IS NULL) AS without_embedding
        FROM kb_chunks
        """
    )
    return {
        "total": stats["total"],
        "without_embedding": stats["without_embedding"],
        "removed": removed["n"],
        "embedded": embedded,
    }


async def backfill_embeddings(batch_size: int = 32) -> int:
    """Compute vectors for chunks that do not have one.

    Only the missing ones, which is what makes `rebuild` cheap to run: editing
    a single lesson step re-embeds that step, not the corpus. The upsert above
    is what marks a chunk as needing this, by nulling the vector whenever the
    text it was derived from changed.
    """
    pending = await db.query_all(
        "SELECT id, title, content FROM kb_chunks WHERE embedding IS NULL ORDER BY id"
    )
    if not pending:
        return 0

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [embedding.document_text(r["title"], r["content"]) for r in batch]
        vectors = await embedding.embed_documents(texts)
        for row, vector in zip(batch, vectors, strict=True):
            await db.execute(
                "UPDATE kb_chunks SET embedding = %s WHERE id = %s",
                (str(vector), row["id"]),
            )

    return len(pending)


async def _main() -> None:
    await db.pool.open()
    await db.pool.wait(timeout=10)
    try:
        stats = await rebuild()
        print(
            f"{stats['total']} chunks | {stats['embedded']} newly embedded | "
            f"{stats['without_embedding']} still without a vector | "
            f"{stats['removed']} removed"
        )
    finally:
        await db.pool.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
