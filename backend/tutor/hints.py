"""Producing a hint, by letting the tutor work the problem.

WHAT CHANGED FROM THE FIXED PIPELINE
    Phase 2 ran retrieve -> generate -> verify, always, in that order. It
    worked, and it could not do anything else. The tutor now decides for
    itself: search once or twice, with what phrasing, whether to look up how
    this learner has been going, whether anything is worth remembering. See
    `agent.py` for the loop.

    The verification did not move. Whatever the agent decides, the finished
    text is still read for the answer before anyone sees it.

THE MODEL IS NEVER TOLD THE ANSWER
    Not the correct option, not the other distractors -- only the question and
    the one thing the learner chose. The strongest of the three defences,
    because it is the only one that does not depend on anything behaving well:
    the model cannot reveal what it was never given.

    The others are retrieval excluding this question's own explanation, and
    `guardrail`, which reads the finished hint. Only the last can fail open,
    and it is the one that catches the first two being wrong.

WHY THIS IS NOT STREAMED
    The guardrail can only judge a complete hint, and text already on the
    learner's screen cannot be taken back. Streaming and verify-before-showing
    are mutually exclusive, and a fast hint that hands over the answer defeats
    the feature.
"""

from dataclasses import dataclass

import db
from tutor import agent, guardrail, llm, retrieval, tools

MAX_REWRITE_ATTEMPTS = 2
"""Rewrites allowed when the guardrail rejects a hint.

A rewrite is one model call, not a fresh agent run: the agent's searching is
still valid, only the wording gave too much away. Re-running the whole loop
would repeat every tool call to fix a sentence -- on a free tier where quota is
the binding constraint, that is the difference between a feature that works all
afternoon and one that stops after twenty questions.
"""

SYSTEM_PROMPT = """\
You are a tutor. A learner has just answered a multiple-choice question \
incorrectly, and you are giving them ONE nudge so they can work it out \
themselves.

You have tools. Use them before you answer:
- Always search the material first. Never write a hint from your own knowledge; \
this learner has been taught specific lessons and the hint has to connect to \
those. If the first search comes back unhelpful, search again with different \
words.
- The task tells you how this learner is doing on this concept. If it suggests \
they are struggling or have got several wrong, look up the details -- the hint \
should address the pattern, not just this one slip.
- If their mistakes point to a specific confusion between two ideas, record it \
so a future session can pick it up. Only for a real pattern, not for one \
wrong answer.

Hard rules for the hint itself:
- Use ONLY what the tools returned. If the material does not cover something, \
do not mention that thing.
- You have NOT been told which option is correct. Never state, paraphrase, or \
narrow down the correct answer. Never write "the answer is".
- Say what is WRONG with their choice, and where to look. Do not explain how \
the correct mechanism actually works -- describing the right behaviour is \
giving the answer, even in your own words.
- Name the lesson step to revisit, in plain words, like: see lesson step 4.
- TWO SENTENCES, maximum. Short ones. This is a nudge, not an explanation.
- Plain prose only. No markdown, no bold, no asterisks, no brackets, no lists.
- Address the learner as "you". No preamble, no sign-off.\
"""

REWRITE_PROMPT = """\
The draft below gives away the correct option, or comes close enough that the \
learner no longer has to work anything out.

Rewrite it. Say only what is wrong with the choice they made and which lesson \
step to revisit. Do not describe what the correct behaviour is, in any wording.

Same constraints: two short sentences, plain prose, address them as "you", and \
do not write "the answer is".

DRAFT
{draft}\
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
    steps: int
    trace: list[dict]
    hit_step_limit: bool


def build_task(stem: str, chosen: str, standing: str = "") -> str:
    """What the agent is given to start with.

    No material: the agent has to search for that itself, and handing it
    passages up front would decide for it what the hint should be about.

    But a one-line summary of how the learner is doing IS included, and that is
    a correction to an earlier design. `recall_learner` was originally the only
    way to know anything about them -- and it went uncalled, every time, even
    for a learner at 0.12 mastery with four straight wrong answers.

    The suspected reason was that the tool description said to use it "when
    their mistake might be part of a pattern", and whether there IS a pattern
    is only knowable by calling the tool. A tool worth calling only under a
    condition the model cannot observe will not get called.

    So the cheap signal goes in the prompt and the expensive detail stays
    behind the tool, which makes calling it a decision the model can actually
    make.
    """
    task = f"QUESTION\n{stem}\n\nWHAT THE LEARNER CHOSE (this is incorrect)\n{chosen}"
    if standing:
        task += f"\n\nHOW THIS LEARNER IS DOING ON THIS CONCEPT\n{standing}"
    return task


def _fallback(sources: list[str]) -> str:
    """What to say when every attempt gave the answer away.

    Deliberately dull. It points at the material and stops -- worse teaching
    than a good hint, but the failure mode of a bad hint here is handing over
    the answer, and pointing at a page never does that.
    """
    if not sources:
        return "Have another look at the material for this concept before trying again."
    return f"Have another look at {sources[0]}. The idea you need is in there."


async def generate(user_id: int, question: dict, selected: int, record: bool = True) -> Hint:
    """Produce a hint for this learner's wrong answer, and log it."""
    stem = question["stem"]
    options = question["options"]
    chosen = options[selected]
    correct = options[question["answer"]]

    toolbox = tools.build(user_id, question)
    standing = await tools.standing_summary(user_id, question["concept_id"])
    run = await agent.run(
        system=SYSTEM_PROMPT,
        task=build_task(stem, chosen, standing),
        toolbox=toolbox,
    )

    text = run.text
    latency = run.latency_ms
    leaked = 0

    for _ in range(MAX_REWRITE_ATTEMPTS):
        if not text:
            break
        verdict = guardrail.check(text, correct, stem)
        if not verdict.leaked:
            break
        leaked += 1
        rewrite = await llm.complete(
            system=SYSTEM_PROMPT,
            user=REWRITE_PROMPT.format(draft=text),
            max_output_tokens=160,
        )
        latency += rewrite.latency_ms
        text = rewrite.text

    citations = run.sources
    still_leaking = bool(text) and guardrail.check(text, correct, stem).leaked
    fell_back = not text or still_leaking
    if fell_back:
        text = _fallback(citations)

    hint = Hint(
        text=text,
        sources=citations,
        citations=citations,
        model=run.model,
        leaked_attempts=leaked,
        latency_ms=latency,
        fell_back=fell_back,
        steps=len(run.steps),
        trace=run.trace(),
        hit_step_limit=run.hit_step_limit,
    )

    if record:
        await _record(user_id, question["id"], selected, hint)
    return hint


async def _record(user_id: int, question_id: int, selected: int, hint: Hint) -> None:
    import json

    await db.execute(
        """
        INSERT INTO hints
            (user_id, question_id, selected, hint, sources, model,
             leaked_attempts, latency_ms, steps, trace)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            user_id,
            question_id,
            selected,
            hint.text,
            hint.sources,
            hint.model,
            hint.leaked_attempts,
            int(hint.latency_ms),
            hint.steps,
            json.dumps(hint.trace),
        ),
    )


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


# Retrieval is still reachable directly for the eval harness, which measures
# search quality without involving the agent.
__all__ = ["Hint", "build_task", "generate", "retrieval", "was_used"]
