# FlowTutor — Flow-State Adaptive Learning Platform

An adaptive learning platform for CS systems design. It keeps every learner in
the flow zone — difficulty always slightly above current ability — with an
agentic AI tutor that scaffolds instead of answering.

**Design principle: optimize for capability, not retention.** The north-star
metric is mastery gain, never streaks or daily active use.

## Status

**Phase 1 complete** — the engine now estimates what each learner knows and
pitches the next question at the edge of it.

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Skeleton: React + FastAPI + Postgres, one question end to end | done |
| 1 | Adaptive difficulty engine (mastery tracking, flow-zone selection) | done |
| 2 | AI tutor: hybrid retrieval + RAG hints | |
| 3 | Agent loop, tools, code sandbox, cross-session memory | |
| 4 | Evals, observability, load testing, deploy | |

## Stack

React 19 + TypeScript (Vite) · FastAPI (async) · PostgreSQL 17 + pgvector ·
Redis · Docker · uv · Alembic · LLM API

Notable choices and why:

- **Async all the way down.** Every route is `async def` over an async psycopg
  pool. Sync routes run in a 40-thread pool; from Phase 2 the tutor holds
  requests open for seconds on LLM calls, which would saturate it.
- **Raw SQL, no ORM.** The Phase 1 selection query and the Phase 2 hybrid
  retrieval query are the core logic of this project; an ORM would hide them.
  SQLAlchemy is present only because Alembic runs on it.
- **pgvector, not a dedicated vector database.** Hybrid retrieval needs keyword
  search too, and Postgres has full-text built in — so both halves live in one
  database and one query.
- **The adaptive model is pure functions.** `backend/adaptive/model.py` has no
  database and no I/O, so it can be unit tested and simulated directly. The
  database layer wraps it rather than being tangled through it.

## The adaptive engine

The product's core loop. Full commentary is in `backend/adaptive/model.py`; the
short version:

**Estimating.** Each learner has a mastery score per concept, on 0..1. After
every answer it moves by the *prediction error* — how surprising the outcome
was, given what the model expected:

    delta = learning_rate * (actual_outcome - predicted_probability)

This is the Elo update rule, and the online form of a 1-parameter item response
model. It replaces "correct: +0.1, wrong: −0.1", which ignores the question: at
mastery 0.5, passing a difficulty-1 question is worth **+0.041** while passing a
difficulty-5 question is worth **+0.184**, and failing the easy one costs
**−0.259** against **−0.116** for the hard one. No special cases produce that —
it falls out of the error term.

Two corrections on top:

- **A guessing floor of 0.25.** Four options means a learner who knows nothing
  still scores 25%. Without it, a lucky guess on a hard question reads as
  mastery.
- **A learning rate that decays with evidence.** The first answer on a concept
  is most of what is known about the learner; the fiftieth is noise.

**Selecting.** Two stages. *Which concept* is sampled with weight
`(1 − mastery)²`, so a lone weak concept dominates the draw while a learner who
is weak everywhere gets their topics interleaved. *How hard* is the difficulty
at which the learner is predicted to succeed 75% of the time — assessment
systems target 50% because a coin flip is maximally informative, but this is a
learning product and the training literature puts optimal learning nearer 80%.

Exploration fires 20% of the time (after a warmup) and ignores both stages.
Without it the engine is a closed loop that only ever confirms its own estimate.

**Known limitation.** Five difficulty rungs only resolve mastery between roughly
0.37 and 0.87. Outside that band the selector clamps and the estimate drifts to
the ceiling or floor. Widening it needs harder and easier *content*, not a
better model. `tests/test_adaptive_model.py` asserts this so it stays visible.

## Quickstart

Three processes: database, backend, frontend.

**1. Infrastructure** (Docker Desktop must be running)

```bash
docker compose up -d
```

Postgres on `localhost:5433`, Redis on `localhost:6380`.

**2. Backend**

```bash
cd backend && uv sync && uv run alembic upgrade head
```

```bash
docker exec -i flowtutor-db psql -U flowtutor -d flowtutor < backend/seed.sql
```

```bash
cd backend && uv run uvicorn main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

**3. Frontend**

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

## Development

```bash
cd backend && uv run pytest -q
```

```bash
cd backend && uv run ruff check . && uv run ruff format .
```

```bash
cd frontend && npm run typecheck
```

## Database migrations

The schema is owned by Alembic, not by a bootstrap script — so adding the
Phase 1 `mastery` table never means wiping the database.

```bash
cd backend && uv run alembic upgrade head
```

```bash
cd backend && uv run alembic revision -m "add mastery table"
```

```bash
cd backend && uv run alembic current
```

`seed.sql` is idempotent (every statement ends in `ON CONFLICT DO NOTHING`), so
reloading content after a migration is safe.

## Layout

```
.
├── docker-compose.yml     Postgres 17 (pgvector) + Redis
├── backend/               FastAPI (async)
│   ├── main.py            API endpoints
│   ├── db.py              async connection pool
│   ├── adaptive/
│   │   ├── model.py       the mathematics -- pure functions, no I/O
│   │   └── engine.py      selection and persistence around it
│   ├── seed.sql           5 concepts, 22 questions (idempotent)
│   ├── alembic/versions/  schema migrations
│   └── tests/             pytest
└── frontend/              React 19 + TypeScript (Vite)
    └── src/
        ├── App.tsx
        ├── api.ts
        ├── types.ts       mirrors the backend response models
        └── components/
            ├── QuestionCard.tsx
            └── MasteryPanel.tsx
```

## API

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | liveness + database connectivity |
| GET | `/concepts` | list concepts |
| GET | `/question?user_id=` | next question, chosen adaptively, **never** includes the answer |
| POST | `/answer` | grades server-side, logs the attempt, updates mastery |
| GET | `/mastery?user_id=` | capability across every concept |
| GET | `/debug/selection?user_id=` | why the engine picks what it picks (development aid) |

**Security invariant:** the correct answer never reaches the client before
submission, and grading always happens on the server. `QuestionOut` has no
`answer` field, so FastAPI strips it even if a query starts selecting `q.*`.
`tests/test_api.py::test_question_never_leaks_the_answer` guards this.

## Performance

Measured numbers are NOT yet valid for the Phase 4 targets. What has been
measured so far, on a development laptop running the database, the dev server
(`--reload`, one worker) and the load generator all at once:

| Measurement | Value | Caveat |
| --- | --- | --- |
| Single-request latency, `/question` | p50 ~2.5 ms | sequential, no contention |
| Throughput, 200 concurrent (ApacheBench) | ~2400 req/s | dev server, same machine |
| p99, 200 concurrent | ~590 ms | queueing, not query time |
| Errors | 0 | `ab` "Failed" counts are length mismatches from variable-size responses |

Phase 4 needs a real load test (k6 or Locust, multiple workers, no `--reload`,
load generator on a separate host) before any of these go anywhere.

## Resetting the database

```bash
docker compose down -v && docker compose up -d
```

Then re-run the migration and seed steps above.
