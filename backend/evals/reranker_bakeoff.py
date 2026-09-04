"""Pick the reranker by measurement, on the same golden set.

Non-commercial licences are excluded regardless of score: this is a portfolio
project that has to stay usable.

Run: uv run python -m evals.reranker_bakeoff
"""

import asyncio
import time

import db
from evals import retrieval_eval
from tutor import retrieval

CANDIDATES = [
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-turbo-en",
    "BAAI/bge-reranker-base",
]


async def main() -> None:
    await db.pool.open()
    await db.pool.wait(timeout=10)
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        queries = retrieval_eval.load_queries()

        # The candidate pool is identical for every reranker, so the comparison
        # measures reranking alone rather than retrieval variance.
        pools = [await retrieval.hybrid_search(c["query"], retrieval.RERANK_POOL) for c in queries]

        print(f"{len(queries)} queries, pool of {retrieval.RERANK_POOL}\n")
        header = f"{'reranker':<40}{'Recall@5':>10}{'Hit@5':>8}{'MRR':>8}{'ms/query':>10}"
        print(header)
        print("-" * len(header))

        for name in CANDIDATES:
            model = TextCrossEncoder(model_name=name)
            recalls, hits, rr = [], [], []
            t0 = time.perf_counter()

            for case, pool in zip(queries, pools, strict=True):
                scores = list(model.rerank(case["query"], [c.content for c in pool]))
                ranked = [
                    c for c, _ in sorted(zip(pool, scores, strict=True), key=lambda p: -p[1])
                ][:5]
                expected = set(case["relevant"])
                top = [c.key for c in ranked]
                overlap = expected & set(top)
                recalls.append(len(overlap) / len(expected))
                hits.append(1.0 if overlap else 0.0)
                rank = next((i + 1 for i, k in enumerate(top) if k in expected), None)
                rr.append(1.0 / rank if rank else 0.0)

            ms = (time.perf_counter() - t0) / len(queries) * 1000
            n = len(queries)
            print(
                f"{name:<40}{sum(recalls) / n:>10.3f}{sum(hits) / n:>8.3f}"
                f"{sum(rr) / n:>8.3f}{ms:>10.1f}"
            )
    finally:
        await db.pool.close()


if __name__ == "__main__":
    asyncio.run(main())
