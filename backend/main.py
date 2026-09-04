"""FastAPI entrypoint -- Phase 0.

Endpoints:
    GET  /health    liveness + database connectivity
    GET  /concepts  list concepts (useful while seeding/debugging)
    GET  /question  serve one question, WITHOUT the answer
    POST /answer    grade a submission server-side and log the attempt

SECURITY INVARIANT (holds for every later phase too):
    The correct answer never leaves the server before the learner has submitted.
    Grading happens here, never in the browser. The client is untrusted input,
    always -- anyone can open devtools and read whatever we send it.

Every route is `async def`. See db.py for why that is not optional here.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.pool.open()
    await db.pool.wait(timeout=10)
    yield
    await db.pool.close()


app = FastAPI(title="FlowTutor API", version="0.1.0", lifespan=lifespan)

# The Vite dev server proxies /api to this process, so CORS is not strictly
# needed in development. Kept for direct calls from the browser or curl.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------- models


class QuestionOut(BaseModel):
    """What the client is allowed to see BEFORE answering.

    Note what is absent: `answer`. Declaring an explicit response_model means
    FastAPI strips any field not listed here, so the answer cannot leak by
    accident even if a future query starts selecting `q.*`.
    """

    id: int
    concept_id: int
    concept_name: str
    stem: str
    options: list[str]
    difficulty: int


class AnswerIn(BaseModel):
    user_id: int = Field(ge=1)
    question_id: int = Field(ge=1)
    selected: int = Field(ge=0)
    time_spent: int | None = Field(default=None, ge=0)


class AnswerOut(BaseModel):
    """What the client sees AFTER submitting -- now the answer is fair game."""

    is_correct: bool
    correct_answer: int
    explanation: str | None
    attempt_id: int


# ------------------------------------------------------------------ endpoints


@app.get("/health")
async def health():
    try:
        await db.query_one("SELECT 1 AS ok")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unreachable: {exc}") from exc
    return {"status": "ok", "database": "ok"}


@app.get("/concepts")
async def list_concepts():
    return await db.query_all("SELECT id, name, description FROM concepts ORDER BY id")


@app.get("/question", response_model=QuestionOut)
async def get_question(user_id: int = Query(default=1, ge=1)):
    """Serve the next question.

    PHASE 0: uniformly random, with one small courtesy -- never repeat the
    question the learner just answered.

    PHASE 1 REPLACES THIS ENTIRE BODY. This is the seam where the adaptive
    engine plugs in: instead of random, it will read the learner's mastery per
    concept and pick a difficulty just above it (the flow zone), with ~20%
    exploration. Everything else in the request path stays the same.
    """
    row = await db.query_one(
        """
        SELECT q.id, q.concept_id, c.name AS concept_name,
               q.stem, q.options, q.difficulty
        FROM questions q
        JOIN concepts c ON c.id = q.concept_id
        WHERE q.id IS DISTINCT FROM (
            SELECT question_id FROM attempts
            WHERE user_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        )
        ORDER BY random()
        LIMIT 1
        """,
        (user_id,),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="no questions available")
    return row


@app.post("/answer", response_model=AnswerOut)
async def submit_answer(payload: AnswerIn):
    """Grade a submission and record the attempt.

    The client sends only which option it picked. We look the truth up here.
    """
    question = await db.query_one(
        "SELECT id, options, answer, explanation FROM questions WHERE id = %s",
        (payload.question_id,),
    )
    if question is None:
        raise HTTPException(status_code=404, detail="question not found")

    if payload.selected >= len(question["options"]):
        raise HTTPException(status_code=400, detail="selected option out of range")

    is_correct = payload.selected == question["answer"]

    attempt = await db.execute(
        """
        INSERT INTO attempts (user_id, question_id, selected, is_correct, time_spent)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            payload.user_id,
            payload.question_id,
            payload.selected,
            is_correct,
            payload.time_spent,
        ),
    )

    # PHASE 1 HOOK: this is where mastery gets updated, right after the attempt
    # is durably logged.

    return AnswerOut(
        is_correct=is_correct,
        correct_answer=question["answer"],
        explanation=question["explanation"],
        attempt_id=attempt["id"],
    )
