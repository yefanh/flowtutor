# FlowTutor — Flow-State Adaptive Learning Platform

An adaptive learning platform for CS systems design. It keeps every learner in
the flow zone — difficulty always slightly above current ability — with an
agentic AI tutor that scaffolds instead of answering.

**Design principle: optimize for capability, not retention.** The north-star
metric is mastery gain, never streaks or daily active use.

## Status

**Teaching mode piloted on one concept** — a learner starting from zero is now
taught before being tested.

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Skeleton: React + FastAPI + Postgres, one question end to end | done |
| 1 | Adaptive difficulty engine (mastery tracking, flow-zone selection) | done |
| 1.5 | Teaching mode — lessons before questions (piloted on Caching) | done |
| 2a | Knowledge base, keyword retrieval, measurement harness | done |
| 2b | Embeddings, hybrid retrieval, reranking | done |
| 2c | Hint generation + hint-before-reveal answer flow | next |
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

## Retrieval

The tutor has to ground its hints in real material, and retrieval quality is the
ceiling on hint quality — a tutor cannot explain using something it never found,
and it will invent something to fill the gap.

**The corpus is assembled, not authored.** Chunks come from material that
already had to exist: lesson steps, and the explanation attached to each
question. The lesson steps were written one idea per step, which is exactly the
unit a retrieval chunk wants to be, so chunking — normally the fiddliest part of
building a RAG corpus — is a no-op here.

**A question never retrieves its own explanation.** That explanation is a
restatement of the answer; handing it to the tutor is handing over the answer to
give away. A prompt instruction not to reveal the answer is a request. Not
putting the answer in front of the model is a guarantee. Both are used.

**Keyword search uses OR, not AND.** Postgres helpers (`websearch_to_tsquery`,
`plainto_tsquery`) join every term with AND, so
`why does my cache still show the old value after an update` becomes
`cach & still & show & old & valu & updat` — and no chunk contains all six.
Measured: every test query returned zero rows before this was changed. Real
ranked retrieval matches any term and ranks by overlap, so query text is run
through `to_tsvector` and the lexemes rejoined with `|`.

### Measured quality

24 golden queries over 29 chunks, phrased the way a stuck learner would put
them. `uv run python -m evals.retrieval_eval`

| Method | Recall@5 | vs keyword | Hit@5 | MRR | queries with nothing relevant |
| --- | --- | --- | --- | --- | --- |
| keyword only | 0.806 | — | 0.875 | 0.651 | 3 |
| dense only | 0.882 | +9.5% | 0.917 | 0.842 | 2 |
| hybrid (RRF) | 0.847 | +5.2% | 0.875 | 0.823 | 3 |
| **hybrid + rerank** | **0.924** | **+14.7%** | **1.000** | **0.892** | **0** |

Hit@5 of 1.000 is the line that matters most: every query now retrieves
something genuinely relevant, so no hint is generated with nothing behind it.

**Two findings that contradict the plan, recorded rather than smoothed over.**

*Hybrid alone is worse than dense alone* (0.847 against 0.882). RRF weights both
methods equally, and keyword search is the weaker of the two here, so fusing it
in at equal weight pulls the ordering down. On 24 queries a gap that size is
within noise, but it does not support the assumption that hybrid always beats
its parts.

What hybrid is actually for, then, is not the final ordering — it is the
candidate pool. Fusion puts the right chunk *somewhere* in the top 10 more
reliably than either method alone, and the reranker fixes the order. That is why
`hybrid + rerank` (0.924) beats `dense + rerank` would-be alternatives while
`hybrid` alone does not beat `dense` alone.

*The target improvement was not met.* The build spec asked for hybrid+rerank to
beat pure dense retrieval by 15–20% on Recall@5. The real number is **+4.8%**
(0.882 → 0.924). Against keyword-only it is +14.7%. The corpus is 29 clean,
well-separated chunks, which leaves little headroom — the honest number stays
here rather than the target.

### Choosing the models

Both by measurement, and both times the smaller model won.

**Embeddings** — `evals/embedding_bakeoff.py`

| Model | dim | Recall@5 | MRR | ms/query |
| --- | --- | --- | --- | --- |
| bge-small-en-v1.5 | 384 | 0.882 | 0.821 | 9 |
| **bge-base-en-v1.5** | **768** | **0.882** | **0.842** | **37** |
| bge-large-en-v1.5 | 1024 | 0.847 | 0.814 | 41 |

The largest was the worst. "Bigger model" is a hypothesis, not a plan.

**Reranker** — `evals/reranker_bakeoff.py`

| Model | Recall@5 | MRR | ms/query |
| --- | --- | --- | --- |
| **ms-marco-MiniLM-L-6-v2** | **0.924** | **0.889** | **255** |
| ms-marco-MiniLM-L-12-v2 | 0.924 | 0.896 | 512 |
| bge-reranker-base | 0.903 | 0.904 | 1923 |
| jina-reranker-v1-turbo-en | 0.882 | 0.814 | 347 |

`jina-reranker-v2` scores higher than any of these and is excluded anyway: it is
CC-BY-NC, and this project has to stay usable.

**Search depth and rerank pool are separate numbers.** Searching deeper is
nearly free; reranking deeper is not — the cross-encoder is the slowest step and
its cost is linear in pool size. Quality plateaus at 10 candidates while latency
keeps climbing (131ms at 10, 260ms at 20, for identical scores). Collapsing both
into one constant shrank the search depth too and dropped Recall@5 from 0.924 to
0.882, which is how the distinction was found.

**Everything runs locally.** Embeddings and reranking are small ONNX models on
the machine: no key, no rate limit, no network round trip, and nothing to pay.
At this corpus size a hosted embedding API would buy nothing.

## Teaching mode

Phase 1 shipped an engine that assumed the learner had studied the material
somewhere else and only needed calibrated practice. For a beginner that fails,
and it fails in a way the engine could not see:

> Mastery alone cannot tell two very different learners apart. One studied the
> concept and scores badly. One has never been shown the concept at all. Both
> sit at the floor of the scale, and both get handed difficulty-1 questions
> forever — but only the first of those is a practice problem.

So `lesson_progress` gives the engine a second input besides the score: has
this learner actually been through the material? `backend/adaptive/teaching.py`
holds the logic. Three rules, in order:

1. **Just finished a lesson and not yet shown you can use it?** Practise that
   concept. Learn one thing, use it, then learn the next.
2. **Something below the teaching threshold with an unread lesson?** Teach the
   next step of it.
3. **Otherwise**, the adaptive practice loop from Phase 1.

Rule 1 is load-bearing. Without it every concept starts below the threshold, so
the engine walks the learner through every lesson of every concept — dozens of
steps of prose — before asking a single question.

The threshold sits at 0.35, which is where Phase 1 *measured* the difficulty
bank running out of road. Below that, the selector clamps to difficulty 1 and
every question is one the learner is expected to fail. That boundary was
reported as a limitation of the practice loop; it is really the line where a
different mechanism belongs.

**Reading a lesson awards no mastery.** It unlocks practice. Reading is not
evidence of being able to do anything, and paying out score for the act of
reading would reward activity instead of capability — the exact trap this
product exists to avoid. Only answers move the number.

Authoring lesson steps for a concept is what switches teaching mode on for it,
so content can be written one concept at a time with no engine change. Lessons
live in `backend/lessons.sql`; only Caching has content so far.

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
docker exec -i flowtutor-db psql -U flowtutor -d flowtutor < backend/lessons.sql
```

```bash
cd backend && uv run python -m tutor.knowledge_base
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

`seed.sql` and `lessons.sql` are both idempotent, so reloading content after a
migration is safe. `lessons.sql` upserts, so revising a lesson step is a matter
of editing the file and applying it again.

## Layout

```
.
├── docker-compose.yml     Postgres 17 (pgvector) + Redis
├── backend/               FastAPI (async)
│   ├── main.py            API endpoints
│   ├── db.py              async connection pool
│   ├── adaptive/
│   │   ├── model.py       the mathematics -- pure functions, no I/O
│   │   ├── engine.py      question selection and persistence
│   │   └── teaching.py    when to explain instead of test
│   ├── tutor/
│   │   ├── knowledge_base.py  assembling the retrievable corpus
│   │   ├── embedding.py       local vectors (bge-base, ONNX)
│   │   ├── reranking.py       local cross-encoder
│   │   └── retrieval.py       keyword, dense, hybrid, rerank
│   ├── evals/
│   │   ├── golden_retrieval.json  24 hand-written query/answer pairs
│   │   ├── retrieval_eval.py      Recall@k, Hit@k, MRR
│   │   ├── embedding_bakeoff.py   picking the embedding model
│   │   └── reranker_bakeoff.py    picking the reranker
│   ├── lessons.sql        lesson content (idempotent)
│   ├── seed.sql           5 concepts, 22 questions (idempotent)
│   ├── alembic/versions/  schema migrations
│   └── tests/             pytest
└── frontend/              React 19 + TypeScript (Vite)
    └── src/
        ├── App.tsx
        ├── api.ts
        ├── types.ts       mirrors the backend response models
        └── components/
            ├── LessonCard.tsx
            ├── QuestionCard.tsx
            └── MasteryPanel.tsx
```

## API

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | liveness + database connectivity |
| GET | `/concepts` | list concepts |
| GET | `/next?user_id=` | what to do now: a lesson step, or a question |
| POST | `/lesson/complete` | mark a lesson step as read (awards no mastery) |
| GET | `/question?user_id=` | next question only, bypassing the teaching check |
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
