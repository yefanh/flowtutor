"""Turning text into vectors, locally.

WHY LOCAL AND NOT A HOSTED API
    The corpus is tens of chunks, not millions. At that size a hosted
    embedding API buys nothing and costs a network round trip on every query,
    an API key, a rate limit, and a dependency that has to be up. Running a
    small model on the machine is faster (no round trip), free, unmetered, and
    works offline.

WHICH MODEL, AND HOW IT WAS CHOSEN
    By measurement, not by size. `evals/embedding_bakeoff.py` scores candidates
    on the same golden set the rest of retrieval is judged by:

        model                dim   Recall@5   MRR   query ms
        bge-small-en-v1.5    384      0.882  0.821       9.1
        bge-base-en-v1.5     768      0.882  0.842      36.5
        bge-large-en-v1.5   1024      0.847  0.814      41.0

    The largest model was the worst. On a small, clean corpus the extra
    capacity has nothing to do, and it is a useful reminder that "bigger model"
    is a hypothesis, not a plan. bge-base wins on the tie-break (best MRR,
    never worse on anything); 36ms is irrelevant next to the LLM call that
    follows it.

    Differences this small on 24 queries are within noise. The finding that
    survives is the negative one: paying 17x the disk and 4.5x the latency for
    bge-large buys nothing here.
"""

import asyncio
from functools import lru_cache

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DIMENSIONS = 768


@lru_cache(maxsize=1)
def _model():
    """Load once, on first use.

    Not at import time: loading takes a couple of seconds and would make every
    process that merely imports this module -- tests, migrations, the eval
    harness -- pay for it whether or not it embeds anything.
    """
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=MODEL_NAME)


def document_text(title: str | None, content: str) -> str:
    """The exact string that gets embedded for a chunk.

    One function so indexing and any future re-indexing cannot disagree about
    the format. Embedding `title + content` at index time but only `content`
    later is a silent, hard-to-spot quality regression.
    """
    return f"{title}\n\n{content}" if title else content


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Vectors for corpus text."""
    if not texts:
        return []
    return await asyncio.to_thread(lambda: [v.tolist() for v in _model().embed(texts)])


async def embed_query(text: str) -> list[float]:
    """Vector for a search query.

    NOT the same call as embed_documents. BGE models are trained asymmetrically:
    queries get an instruction prefix that documents do not. `query_embed`
    applies it. Using the document path for a query still returns a plausible
    vector and silently retrieves worse -- there is no error to notice.

    Runs in a thread: this is CPU-bound work, and the whole backend is async so
    that no request can stall the event loop for other requests.
    """
    return await asyncio.to_thread(lambda: next(iter(_model().query_embed([text]))).tolist())
