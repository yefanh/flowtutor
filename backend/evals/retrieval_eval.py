"""Measuring retrieval quality against the golden set.

WHY THIS EXISTS BEFORE THE RETRIEVAL IT MEASURES
    Retrieval quality is the ceiling on hint quality -- the tutor cannot
    explain using material it never found. Every improvement from here on
    (embeddings, hybrid search, reranking) is a claim that something got
    better, and without a number those claims are decoration. Building the
    ruler first also means the first number is an honest baseline rather than
    something produced after the fact to flatter a finished system.

THE METRICS
    Recall@k   Of the chunks that genuinely answer this query, what fraction
               made it into the top k? This is the headline number: material
               that is not retrieved cannot be used.
    Hit@k      Did at least one relevant chunk make the top k? A query that
               scores 0 here produced a hint with nothing behind it.
    MRR        1 / rank of the first relevant chunk, averaged. Sensitive to
               ordering in a way recall is not -- with a limited prompt budget,
               rank 1 and rank 5 are not equally useful.

Run it:  uv run python -m evals.retrieval_eval
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import db
from tutor import retrieval

GOLDEN_SET = Path(__file__).parent / "golden_retrieval.json"

# A retrieval method: query text in, ranked chunk keys out.
Method = Callable[[str, int], Awaitable[list[str]]]


async def _keyword(query: str, limit: int) -> list[str]:
    return [c.key for c in await retrieval.keyword_search(query, limit=limit)]


# Adding a method is one entry here; every metric and comparison follows.
METHODS: dict[str, Method] = {
    "keyword": _keyword,
}


@dataclass(frozen=True)
class Result:
    method: str
    k: int
    recall: float
    hit_rate: float
    mrr: float
    misses: list[str]


def load_queries() -> list[dict]:
    return json.loads(GOLDEN_SET.read_text())["queries"]


async def validate_golden_set() -> list[str]:
    """Every key the golden set points at must exist in the corpus.

    A typo'd key is indistinguishable from a retrieval failure in the scores,
    which would quietly drag every number down and send you debugging the
    wrong thing.
    """
    referenced = {key for q in load_queries() for key in q["relevant"]}
    rows = await db.query_all("SELECT key FROM kb_chunks")
    existing = {r["key"] for r in rows}
    return sorted(referenced - existing)


async def evaluate(name: str, method: Method, k: int = 5) -> Result:
    queries = load_queries()
    recalls, hits, reciprocal_ranks, misses = [], [], [], []

    for case in queries:
        expected = set(case["relevant"])
        found = await method(case["query"], k)
        found_set = set(found)

        overlap = expected & found_set
        recalls.append(len(overlap) / len(expected))
        hits.append(1.0 if overlap else 0.0)

        rank = next((i + 1 for i, key in enumerate(found) if key in expected), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        if not overlap:
            misses.append(case["query"])

    n = len(queries)
    return Result(
        method=name,
        k=k,
        recall=sum(recalls) / n,
        hit_rate=sum(hits) / n,
        mrr=sum(reciprocal_ranks) / n,
        misses=misses,
    )


async def run(k: int = 5) -> list[Result]:
    return [await evaluate(name, method, k) for name, method in METHODS.items()]


async def main() -> None:
    await db.pool.open()
    await db.pool.wait(timeout=10)
    try:
        broken = await validate_golden_set()
        if broken:
            print("GOLDEN SET REFERENCES CHUNKS THAT DO NOT EXIST:")
            for key in broken:
                print(f"  {key}")
            print("\nRebuild the knowledge base or fix the keys, then rerun.\n")
            return

        queries = load_queries()
        total = await db.query_one("SELECT count(*) AS n FROM kb_chunks")
        print(f"{len(queries)} queries against {total['n']} chunks\n")

        header = f"{'method':<20}{'Recall@5':>10}{'Hit@5':>8}{'MRR':>8}{'misses':>8}"
        print(header)
        print("-" * len(header))

        results = await run(k=5)
        for r in results:
            print(
                f"{r.method:<20}{r.recall:>10.3f}{r.hit_rate:>8.3f}{r.mrr:>8.3f}{len(r.misses):>8}"
            )

        for r in results:
            if r.misses:
                print(f"\n{r.method} found nothing relevant for:")
                for query in r.misses:
                    print(f"  - {query}")
    finally:
        await db.pool.close()


if __name__ == "__main__":
    asyncio.run(main())
