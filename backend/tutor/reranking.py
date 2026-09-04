"""Cross-encoder reranking, locally.

Same reasoning as embedding: local, free, unmetered, no key. The model is
picked by measurement in `evals/reranker_bakeoff.py`.
"""

import asyncio
from functools import lru_cache

MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=MODEL_NAME)


async def score(query: str, documents: list[str]) -> list[float]:
    """Relevance of each document to the query.

    Scores are unbounded logits, comparable only within one call -- they order
    a shortlist, they do not mean anything on their own.
    """
    if not documents:
        return []
    return await asyncio.to_thread(lambda: list(_model().rerank(query, documents)))
