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
from tutor import agent, hints, llm


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


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the model with a scripted one.

    `replies` is a queue of what the model says next. An entry may be a string
    (it answers) or a list of ToolCall (it asks for tools), which is what makes
    the agent loop testable without a network.

    Returns a recorder so tests can assert on what was sent -- which is where
    the strongest guarantee lives: the correct answer is never in the prompt.
    """
    sent: list[dict] = []
    replies: list = []

    async def fake_complete(
        system: str,
        user: str | None = None,
        max_output_tokens: int = 400,
        turns: list | None = None,
        tools: list | None = None,
    ):
        conversation = "\n".join(t.text for t in turns if t.text) if turns else (user or "")
        sent.append(
            {
                "system": system,
                "text": conversation,
                "tools": [t.name for t in (tools or [])],
                "turns": turns,
            }
        )

        reply = replies.pop(0) if replies else "Have a look at what the lesson says."
        if isinstance(reply, list):
            return llm.Completion(
                text="",
                model="stub",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                tool_calls=tuple(reply),
            )
        return llm.Completion(
            text=reply,
            model="stub",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
        )

    monkeypatch.setattr(llm, "complete", fake_complete)
    monkeypatch.setattr(hints.llm, "complete", fake_complete)
    monkeypatch.setattr(agent.llm, "complete", fake_complete)
    return {"sent": sent, "replies": replies}
