"""Compare embedding models on the golden set.

Scored in memory rather than through pgvector on purpose: each candidate has a
different dimension, so going through the database would mean a migration per
candidate. The winner gets stored; the losers cost nothing.

Result on 29 chunks / 24 queries -- the largest model was the worst:

    model                dim   Recall@5   MRR   query ms
    bge-small-en-v1.5    384      0.882  0.821       9.1
    bge-base-en-v1.5     768      0.882  0.842      36.5
    bge-large-en-v1.5   1024      0.847  0.814      41.0

Run: uv run python -m evals.embedding_bakeoff
"""

import asyncio
import time

import numpy as np

import db
from evals import retrieval_eval

CANDIDATES = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
]


async def main():
    await db.pool.open()
    await db.pool.wait(timeout=10)
    rows = await db.query_all("SELECT key, title, content FROM kb_chunks ORDER BY id")
    keys = [r["key"] for r in rows]
    # Title carries real signal ("What you gave up: the copy can go stale"),
    # so it is embedded together with the body, same as keyword search weights it.
    docs = [f"{r['title']}\n\n{r['content']}" for r in rows]
    queries = retrieval_eval.load_queries()

    from fastembed import TextEmbedding

    print(f"{len(docs)} chunks, {len(queries)} queries\n")
    header = (
        f"{'model':<28}{'dim':>5}{'Recall@5':>10}"
        f"{'Hit@5':>8}{'MRR':>8}{'index s':>9}{'query ms':>10}"
    )
    print(header)
    print("-" * len(header))

    for name in CANDIDATES:
        model = TextEmbedding(model_name=name)
        t0 = time.perf_counter()
        D = np.array(list(model.embed(docs)), dtype=np.float32)
        index_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        Q = np.array(list(model.query_embed([q["query"] for q in queries])), dtype=np.float32)
        query_ms = (time.perf_counter() - t0) / len(queries) * 1000

        D /= np.linalg.norm(D, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        sims = Q @ D.T

        recalls, hits, rr = [], [], []
        for i, case in enumerate(queries):
            expected = set(case["relevant"])
            top = [keys[j] for j in np.argsort(-sims[i])[:5]]
            overlap = expected & set(top)
            recalls.append(len(overlap) / len(expected))
            hits.append(1.0 if overlap else 0.0)
            rank = next((n + 1 for n, k in enumerate(top) if k in expected), None)
            rr.append(1.0 / rank if rank else 0.0)
        n = len(queries)
        print(
            f"{name:<28}{D.shape[1]:>5}{sum(recalls) / n:>10.3f}{sum(hits) / n:>8.3f}"
            f"{sum(rr) / n:>8.3f}{index_s:>9.1f}{query_ms:>10.1f}"
        )

    await db.pool.close()


if __name__ == "__main__":
    asyncio.run(main())
