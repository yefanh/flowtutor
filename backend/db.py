"""Database access: one async connection pool for the whole process.

WHY ASYNC (and not a sync pool behind FastAPI's threadpool):
    FastAPI runs sync (`def`) endpoints in an anyio worker threadpool that
    defaults to 40 threads. That is a hard concurrency ceiling. It is survivable
    for millisecond-scale database queries, but from Phase 2 on, the tutor calls
    an LLM and holds each request open for seconds. Forty concurrent hint
    requests would saturate the pool and block *every* other route, including
    plain question fetches. Async endpoints release the event loop while
    waiting on I/O, so a slow LLM call costs a coroutine, not a thread.

WHY A POOL:
    Opening a Postgres connection costs several milliseconds and a backend
    process on the server. The Phase 4 target is p99 < 200ms at 500 concurrent
    users; per-request connects would spend that budget on setup alone.
"""

import os

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://flowtutor:flowtutor@localhost:5433/flowtutor",
)

# open=False so importing this module does not hit the network. The pool is
# opened explicitly in the FastAPI lifespan handler and closed on shutdown.
pool = AsyncConnectionPool(
    conninfo=DATABASE_URL,
    min_size=1,
    max_size=10,
    open=False,
    kwargs={"row_factory": dict_row},
)


async def query_one(sql: str, params: tuple = ()) -> dict | None:
    """Run a SELECT and return the first row as a dict, or None."""
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchone()


async def query_all(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT and return all rows as dicts."""
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


async def execute(sql: str, params: tuple = ()) -> dict | None:
    """Run an INSERT/UPDATE/DELETE. Returns the RETURNING row if there is one.

    psycopg commits when the `connection()` block exits cleanly, and rolls back
    if it raises.
    """
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchone() if cur.description else None
