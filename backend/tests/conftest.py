"""Shared test fixtures.

NOTE: these tests run against the development database, not a throwaway one.
That is acceptable while the only writes are appends to `attempts`, but it
should become a dedicated test database (or a transaction rolled back per test)
before Phase 4 wires this into CI.
"""

import httpx
import pytest

import db
from main import app


@pytest.fixture(scope="session", autouse=True)
async def _pool():
    """Open the connection pool once for the whole session.

    psycopg pools cannot be reopened after closing, so this is session-scoped
    rather than per-test.
    """
    await db.pool.open()
    await db.pool.wait(timeout=10)
    yield
    await db.pool.close()


@pytest.fixture
async def client():
    """An HTTP client that talks to the app in-process (no network, no server)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
