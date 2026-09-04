"""Generating a hint: retrieve, ground, generate, verify.

WHAT A HINT IS FOR
    The learner has answered wrong and has not been shown the right answer. The
    hint exists so they can get there themselves. It should engage with the
    specific mistake they made, point at the material that resolves it, and
    stop.

THE MODEL IS NEVER TOLD THE ANSWER
    Not the correct option, not the other options -- only the question and the
    one thing the learner chose. This is the strongest of the three defences
    against giving the answer away, because it is the only one that does not
    depend on anything behaving well: the model cannot reveal what it was never
    given.

    The other two are retrieval excluding the question's own explanation (which
    restates its answer), and `guardrail`, which reads the finished hint and
    rejects it if the answer surfaced anyway.

WHY THIS IS NOT STREAMED
    Streaming would show the first words sooner, which is what the latency
    target asks for. But the guardrail can only judge a hint once it is
    complete, and text already on the learner's screen cannot be taken back.
    Streaming and verifying-before-showing are mutually exclusive here, and
    verification is the more important of the two: a fast hint that hands over
    the answer defeats the feature. The UI shows a pending state instead.
"""

from dataclasses import dataclass

import db
from tutor import guardrail, llm, retrieval

MAX_GENERATION_ATTEMPTS = 3
"""How many times to ask again when the guardrail rejects a hint."""

SYSTEM_PROMPT = """\
You are a tutor. A learner has just answered a multiple-choice question \
incorrectly, and you are giving them ONE nudge so they can work it out \
themselves.

Hard rules:
- Use ONLY the reference material provided. If it does not cover something, do \
not mention that thing.
- You have NOT been told which option is correct. Never state, paraphrase, or \
narrow down the correct answer. Never write "the answer is".
- Say what is WRONG with their choice, and where to look. Do not explain how \
the correct mechanism actually works -- describing the right behaviour is \
giving the answer, even in your own words.
- Name the lesson step to revisit, in plain words, like: see lesson step 4.
- TWO SENTENCES, maximum. Short ones. This is a nudge, not an explanation.
- Plain prose only. No markdown, no bold, no asterisks, no brackets, no lists, \
no headings.
- Address the learner as "you". No preamble, no sign-off, no asking whether \
they understood.\
"""

RETRY_SUFFIX = """\

Your previous attempt gave the answer away. Write a different nudge that does \
not name, describe, or restate the correct option at all -- point at where to \
look and what to reconsider instead.\
"""


@dataclass(frozen=True)
class Hint:
    text: str
    sources: list[str]
    citations: list[str]
    model: str | None
    leaked_attempts: int
    latency_ms: float
    fell_back: bool


def build_prompt(stem: str, chosen: str, chunks: list[retrieval.RetrievedChunk]) -> str:
    material = "\n\n".join(f"[{chunk.citation}]\n{chunk.content}" for chunk in chunks)
    return (
        f"QUESTION\n{stem}\n\n"
        f"WHAT THE LEARNER CHOSE (this is incorrect)\n{chosen}\n\n"
        f"REFERENCE MATERIAL\n{material}"
    )


def _fallback(chunks: list[retrieval.RetrievedChunk]) -> str:
    """What to say when every generated hint gave the answer away.

    Deliberately dull. It points at the material and stops -- worse teaching
    than a good hint, but the failure mode of a bad hint here is handing over
    the answer, and pointing at a page never does that.
    """
    if not chunks:
        return "Have another look at the material for this concept before trying again."
    return (
        f"Have another look at {chunks[0].citation}. The idea you need to reconsider is in there."
    )


async def generate(user_id: int, question: dict, selected: int, record: bool = True) -> Hint:
    """Produce a hint for this learner's wrong answer, and log it."""
    stem = question["stem"]
    options = question["options"]
    chosen = options[selected]
    correct = options[question["answer"]]

    # The learner's wrong choice is part of the query, not just the stem: it is
    # what makes the retrieved material about their misconception rather than
    # about the topic in general.
    chunks = await retrieval.search(
        f"{stem} {chosen}",
        concept_id=question["concept_id"],
        exclude_question_id=question["id"],
    )

    prompt = build_prompt(stem, chosen, chunks)
    leaked = 0
    text = ""
    model: str | None = None
    latency = 0.0

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        system = SYSTEM_PROMPT + (RETRY_SUFFIX if attempt else "")
        # Capped low: the brevity rule in the prompt is a request, and a hard
        # ceiling makes a rambling hint impossible rather than unlikely.
        completion = await llm.complete(system=system, user=prompt, max_output_tokens=160)
        latency += completion.latency_ms
        model = completion.model

        verdict = guardrail.check(completion.text, correct, stem)
        if not verdict.leaked:
            text = completion.text
            break
        leaked += 1

    fell_back = not text
    if fell_back:
        text = _fallback(chunks)

    hint = Hint(
        text=text,
        sources=[c.key for c in chunks],
        citations=[c.citation for c in chunks],
        model=model,
        leaked_attempts=leaked,
        latency_ms=latency,
        fell_back=fell_back,
    )

    if record:
        await db.execute(
            """
            INSERT INTO hints
                (user_id, question_id, selected, hint, sources, model,
                 leaked_attempts, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                question["id"],
                selected,
                hint.text,
                hint.sources,
                hint.model,
                hint.leaked_attempts,
                int(hint.latency_ms),
            ),
        )

    return hint


async def was_used(user_id: int, question_id: int) -> bool:
    """Has this learner been given a hint for this question?

    Server-side on purpose. This decides how much mastery a correct answer
    earns, and a client that reported its own hint usage could claim full
    credit for an assisted answer.
    """
    row = await db.query_one(
        "SELECT 1 AS found FROM hints WHERE user_id = %s AND question_id = %s LIMIT 1",
        (user_id, question_id),
    )
    return row is not None
