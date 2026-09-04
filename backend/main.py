"""FastAPI entrypoint -- Phase 1.

Endpoints:
    GET  /health    liveness + database connectivity
    GET  /concepts  list concepts
    GET  /next      the next thing to do: a lesson step, or a question
    GET  /question  serve the next question, chosen adaptively, WITHOUT the answer
    POST /answer    grade a submission, update mastery, log the attempt
    POST /hint      a grounded nudge for a wrong answer, without the answer
    POST /lesson/complete  mark a lesson step as read
    GET  /mastery   this learner's mastery and lesson progress per concept

SECURITY INVARIANT (holds for every later phase too):
    The correct answer never leaves the server before the learner has submitted.
    Grading happens here, never in the browser. The client is untrusted input,
    always -- anyone can open devtools and read whatever we send it.

Every route is `async def`. See db.py for why that is not optional here.
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db
from adaptive import engine, model, teaching
from tutor import hints, llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.pool.open()
    await db.pool.wait(timeout=10)
    yield
    await db.pool.close()


app = FastAPI(title="FlowTutor API", version="0.4.0", lifespan=lifespan)

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


class MasteryDelta(BaseModel):
    """How this answer moved the learner's estimate for the concept.

    Returned so the UI can show capability changing -- the only kind of
    progress feedback this product gives.
    """

    concept_id: int
    concept_name: str
    previous: float
    current: float
    delta: float
    crossed_threshold: bool


class AnswerOut(BaseModel):
    """What the client sees after submitting.

    `correct_answer` and `explanation` are None until the answer is revealed.
    A first wrong attempt deliberately withholds both: the learner gets a hint
    and another try instead. Showing the answer immediately would make the
    hint pointless -- there is nothing to work out once you have been told.
    """

    is_correct: bool
    revealed: bool
    correct_answer: int | None
    explanation: str | None
    attempt_id: int
    mastery: MasteryDelta
    used_hint: bool
    mastery_updated: bool


class HintIn(BaseModel):
    user_id: int = Field(ge=1)
    question_id: int = Field(ge=1)
    selected: int = Field(ge=0)


class HintOut(BaseModel):
    hint: str
    citations: list[str]


class LessonStepOut(BaseModel):
    lesson_id: int
    concept_id: int
    concept_name: str
    step: int
    total_steps: int
    title: str
    body: str


class NextOut(BaseModel):
    """Either a lesson step or a question, tagged so the client can branch.

    A tagged union rather than two endpoints, because deciding WHICH of the two
    a learner needs is the engine's job, not the interface's. If the client had
    to ask "do I need teaching?" first, that decision would leak into the UI
    and drift out of sync with the engine.
    """

    kind: Literal["lesson", "question"]
    lesson: LessonStepOut | None = None
    question: QuestionOut | None = None


class LessonCompleteIn(BaseModel):
    user_id: int = Field(ge=1)
    lesson_id: int = Field(ge=1)


class ConceptMasteryOut(BaseModel):
    concept_id: int
    concept_name: str
    score: float
    attempts: int
    is_mastered: bool
    lesson_steps_total: int
    lesson_steps_done: int


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


@app.get("/next", response_model=NextOut)
async def get_next(user_id: int = Query(default=1, ge=1)):
    """What this learner should do next.

    Three rules, in order:

      1. Just finished a lesson and not yet shown you can use it? Practise
         THAT concept. Learn one thing, use it, then learn the next.
      2. Something below the teaching threshold with an unread lesson? Teach
         the next step of it.
      3. Otherwise, the adaptive practice loop.

    Rule 1 has to come first. Without it the engine reads every lesson of
    every concept before asking anything, because they all start below the
    threshold -- thirty-five steps of prose before a single question.
    """
    awaiting = await teaching.concept_awaiting_practice(user_id)
    if awaiting is not None:
        question = await engine.select_question(user_id, force_concept_id=awaiting)
        if question is not None:
            return NextOut(kind="question", question=QuestionOut(**question))

    step = await teaching.next_lesson_step(user_id)
    if step is not None:
        return NextOut(kind="lesson", lesson=LessonStepOut(**vars(step)))

    question = await engine.select_question(user_id)
    if question is None:
        raise HTTPException(status_code=404, detail="nothing left to do")
    return NextOut(kind="question", question=QuestionOut(**question))


@app.post("/lesson/complete")
async def complete_lesson_step(payload: LessonCompleteIn):
    """Mark a step as read.

    This unlocks practice for the concept. It deliberately does NOT touch
    mastery: reading is not evidence of capability, and paying out score for it
    would reward activity instead of learning.
    """
    if not await teaching.complete_step(payload.user_id, payload.lesson_id):
        raise HTTPException(status_code=404, detail="lesson step not found")
    return {"status": "ok"}


@app.get("/question", response_model=QuestionOut)
async def get_question(user_id: int = Query(default=1, ge=1)):
    """Serve the next question, chosen by the adaptive engine.

    The route stays thin on purpose: all of the selection logic lives in
    adaptive/, where it can be tested and simulated without HTTP or a server.
    """
    question = await engine.select_question(user_id)
    if question is None:
        raise HTTPException(status_code=404, detail="no questions available")
    return question


@app.post("/answer", response_model=AnswerOut)
async def submit_answer(payload: AnswerIn):
    """Grade a submission, record it, and update the mastery estimate.

    The client sends only which option it picked. Everything else -- whether
    the answer is right, whether a hint was used, whether to reveal -- is
    decided here. In particular `used_hint` is read from the database rather
    than taken from the request: it reduces the mastery a correct answer earns,
    so a client that reported its own hint usage could claim full credit for an
    assisted answer.
    """
    question = await _load_question(payload.question_id)

    if payload.selected >= len(question["options"]):
        raise HTTPException(status_code=400, detail="selected option out of range")

    used_hint = await hints.was_used(payload.user_id, payload.question_id)

    result = await engine.apply_attempt(
        user_id=payload.user_id,
        question=question,
        selected=payload.selected,
        time_spent=payload.time_spent,
        used_hint=used_hint,
    )

    # Reveal on success, or once this is no longer their first go at it.
    reveal = result.is_correct or not result.first_attempt

    return AnswerOut(
        is_correct=result.is_correct,
        revealed=reveal,
        correct_answer=result.correct_answer if reveal else None,
        explanation=result.explanation if reveal else None,
        attempt_id=result.attempt_id,
        used_hint=result.used_hint,
        mastery_updated=result.mastery_updated,
        mastery=MasteryDelta(
            concept_id=result.concept_id,
            concept_name=result.concept_name,
            previous=result.mastery_before,
            current=result.mastery_after,
            delta=result.mastery_after - result.mastery_before,
            crossed_threshold=result.crossed_threshold,
        ),
    )


@app.post("/hint", response_model=HintOut)
async def get_hint(payload: HintIn):
    """A nudge for a wrong answer.

    The generator is never told which option is correct, retrieval excludes
    this question's own explanation, and the finished text is checked for the
    answer before it is returned. See tutor/hints.py.
    """
    question = await _load_question(payload.question_id)

    if payload.selected >= len(question["options"]):
        raise HTTPException(status_code=400, detail="selected option out of range")
    if payload.selected == question["answer"]:
        raise HTTPException(status_code=400, detail="that answer was correct")

    try:
        hint = await hints.generate(payload.user_id, question, payload.selected)
    except llm.LLMError as exc:
        # Free-tier quotas are small and the daily one is easy to hit, so this
        # is an expected state rather than a crash. The learner is told to try
        # again rather than shown a stack trace, and the question stays
        # unanswered so nothing is lost.
        raise HTTPException(
            status_code=503,
            detail="The tutor is unavailable right now. Try again in a minute.",
        ) from exc

    return HintOut(hint=hint.text, citations=hint.citations)


async def _load_question(question_id: int) -> dict:
    question = await db.query_one(
        """
        SELECT q.id, q.concept_id, c.name AS concept_name,
               q.stem, q.options, q.answer, q.difficulty, q.explanation
        FROM questions q
        JOIN concepts c ON c.id = q.concept_id
        WHERE q.id = %s
        """,
        (question_id,),
    )
    if question is None:
        raise HTTPException(status_code=404, detail="question not found")
    return question


@app.get("/mastery", response_model=list[ConceptMasteryOut])
async def get_mastery(user_id: int = Query(default=1, ge=1)):
    """This learner's capability across every concept.

    This is the progress surface of the product. It reports what the learner
    can now do -- deliberately not how many days in a row they showed up.
    """
    lessons = {row["concept_id"]: row for row in await teaching.lesson_state(user_id)}
    return [
        ConceptMasteryOut(
            concept_id=c.concept_id,
            concept_name=c.concept_name,
            score=c.score,
            attempts=c.attempts,
            is_mastered=c.is_mastered,
            lesson_steps_total=lessons.get(c.concept_id, {}).get("total_steps", 0),
            lesson_steps_done=lessons.get(c.concept_id, {}).get("completed_steps", 0),
        )
        for c in await engine.get_mastery(user_id)
    ]


@app.get("/debug/hints")
async def debug_hints(user_id: int = Query(default=1, ge=1), limit: int = Query(10, ge=1)):
    """What the tutor did, step by step. Development aid, not product.

    An agent is several decisions deep, so when a hint is wrong the question is
    always WHICH step was wrong -- a bad search, a bad reading of good
    material, or a bad write-up. From the outside those look identical. This is
    the view that tells them apart.
    """
    rows = await db.query_all(
        """
        SELECT h.created_at, h.model, h.steps, h.trace, h.sources,
               h.leaked_attempts, h.latency_ms, h.hint, q.stem,
               q.options ->> h.selected AS they_chose
        FROM hints h
        JOIN questions q ON q.id = h.question_id
        WHERE h.user_id = %s
        ORDER BY h.id DESC
        LIMIT %s
        """,
        (user_id, limit),
    )
    return rows


@app.get("/debug/selection")
async def debug_selection(user_id: int = Query(default=1, ge=1)):
    """Why the engine would pick what it picks. Development aid, not product.

    Shows the target difficulty and predicted success rate per concept, which
    is the fastest way to check the engine is behaving before trusting the UI.
    """
    return [
        {
            "concept": c.concept_name,
            "mastery": round(c.score, 3),
            "attempts": c.attempts,
            "target_difficulty": round(model.target_difficulty(c.score), 2),
            "predicted_success": {
                f"d{d}": round(model.probability_correct(c.score, d), 2) for d in range(1, 6)
            },
        }
        for c in await engine.get_mastery(user_id)
    ]
