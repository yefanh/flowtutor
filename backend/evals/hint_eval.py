"""Measuring hint quality, and answering whether the agent is worth it.

TWO PHASES, KEPT SEPARATE
    Generation writes hints to a cache file. Judging reads that file. They are
    separate commands because they fail differently and cost differently:
    generation burns the free-tier quota and is slow; judging is cheap to
    re-run and is the part that gets iterated on. Rewriting the rubric should
    not mean paying to regenerate every hint.

WHY THE JUDGE IS TOLD THE ANSWER AND THE TUTOR IS NOT
    The spoiler check is the point. `guardrail.py` compares words, so it
    catches quoting and paraphrase-with-shared-vocabulary and nothing else --
    "defers the store write" restated as "delays writing to the database" walks
    straight past it. Deciding whether two sentences mean the same thing needs
    a reader, and a reader needs to know what the answer is.

    So the judge gets what the tutor was denied. That asymmetry is the whole
    design.

WHAT THIS IS NOT
    An LLM grading an LLM is a proxy, not a verdict. It is lenient, it can be
    swayed by fluent writing, and it has no idea what a confused human would
    actually take away from a sentence. Two guards against believing it too
    much: every judgement carries a one-line reason so it can be argued with,
    and `--show` prints hints and verdicts side by side for a human to check.
    A judge nobody has ever disagreed with has not been validated, it has just
    not been read.

Run:
    uv run python -m evals.hint_eval generate            # writes the cache
    uv run python -m evals.hint_eval generate --arm direct
    uv run python -m evals.hint_eval judge
    uv run python -m evals.hint_eval show
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import db
from tutor import agent, guardrail, hints, llm, retrieval, tools

CASES = Path(__file__).parent / "hint_cases.json"
CACHE = Path(__file__).parent / "hint_runs"

JUDGE_MODEL = "gemini-3.1-flash-lite"

RATE_LIMIT_WAIT = 90.0
"""Seconds this harness may sleep when the provider says "too fast".

The free tier caps requests per MINUTE -- 15 on Flash-Lite -- so a ten-case run
hits it almost immediately. In the request path the wait is zero, because a
learner should get a fallback model or an error rather than a 35-second pause.
A batch job has no user waiting and would otherwise die a third of the way in.
"""

JUDGE_SYSTEM = """\
You are grading a hint that a tutor gave to a learner who answered a \
multiple-choice question incorrectly.

You are shown the correct answer. The tutor that wrote the hint was NOT -- it \
only saw the question and the option the learner picked.

Grade four things independently. For each, answer true or false and give one \
short sentence of reasoning.

grounded: every factual claim in the hint is supported by the reference \
material shown. False if it asserts something the material does not say.

targeted: it engages with the specific option this learner chose, rather than \
being generic advice about the topic.

no_spoiler: it does not state, paraphrase, or narrow down to the correct \
answer -- INCLUDING describing the correct mechanism in different words. Be \
strict. Ask yourself: after reading only this hint, would the learner know \
which option to pick? If yes, this is false.

actionable: it gives the learner something concrete to reconsider, or \
somewhere specific to look.\
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "grounded_reason": {"type": "string"},
        "targeted": {"type": "boolean"},
        "targeted_reason": {"type": "string"},
        "no_spoiler": {"type": "boolean"},
        "no_spoiler_reason": {"type": "string"},
        "actionable": {"type": "boolean"},
        "actionable_reason": {"type": "string"},
    },
    "required": [
        "grounded",
        "grounded_reason",
        "targeted",
        "targeted_reason",
        "no_spoiler",
        "no_spoiler_reason",
        "actionable",
        "actionable_reason",
    ],
}

CRITERIA = ("grounded", "targeted", "no_spoiler", "actionable")

DIRECT_SYSTEM = hints.SYSTEM_PROMPT.replace(
    """You have tools. Use them before you answer:
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
""",
    "",
)


@dataclass
class Generated:
    case: dict
    stem: str
    chosen: str
    correct: str
    hint: str
    material: list[str]
    steps: int
    tools_used: list[str]
    model: str | None
    latency_ms: float
    lexical_guardrail_blocked: int


async def _load_question(question_id: int, expected_stem: str) -> dict:
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
    if question is None or not question["stem"].startswith(expected_stem):
        raise SystemExit(
            f"case points at question {question_id}, which is not "
            f"{expected_stem!r}. Reseed, or fix the case file."
        )
    return question


async def _direct(question: dict, selected: int) -> Generated:
    """The Phase 2 pipeline: retrieve, then generate. No choices.

    Kept here rather than in production code because its only purpose is to be
    the control arm. Holding the model fixed and removing the tool choice is
    what isolates what the agent loop actually contributes.
    """
    stem, options = question["stem"], question["options"]
    chunks = await retrieval.search(
        f"{stem} {options[selected]}",
        concept_id=question["concept_id"],
        exclude_question_id=question["id"],
    )
    material = "\n\n".join(f"Source -- {c.citation}\n{c.content}" for c in chunks)
    prompt = (
        f"QUESTION\n{stem}\n\n"
        f"WHAT THE LEARNER CHOSE (this is incorrect)\n{options[selected]}\n\n"
        f"REFERENCE MATERIAL\n{material}"
    )
    completion = await llm.complete(system=DIRECT_SYSTEM, user=prompt, max_output_tokens=320)
    return Generated(
        case={},
        stem=stem,
        chosen=options[selected],
        correct=options[question["answer"]],
        hint=completion.text,
        material=[c.citation for c in chunks],
        steps=0,
        tools_used=[],
        model=completion.model,
        latency_ms=completion.latency_ms,
        lexical_guardrail_blocked=0,
    )


async def _agent(question: dict, selected: int) -> Generated:
    stem, options = question["stem"], question["options"]
    toolbox = tools.build(0, question)
    standing = await tools.standing_summary(0, question["concept_id"])
    run = await agent.run(
        system=hints.SYSTEM_PROMPT,
        task=hints.build_task(stem, options[selected], standing),
        toolbox=toolbox,
        rate_limit_wait=RATE_LIMIT_WAIT,
    )
    return Generated(
        case={},
        stem=stem,
        chosen=options[selected],
        correct=options[question["answer"]],
        hint=run.text,
        material=run.sources,
        steps=len(run.steps),
        tools_used=[s.tool for s in run.steps],
        model=run.model,
        latency_ms=run.latency_ms,
        lexical_guardrail_blocked=0,
    )


async def generate(arm: str, model: str | None) -> None:
    CACHE.mkdir(exist_ok=True)
    cases = json.loads(CASES.read_text())["cases"]
    if model:
        llm.MODEL = model

    path = CACHE / f"{arm}--{(model or llm.MODEL).replace('/', '_')}.json"

    # Written after every case, not at the end. The first run of this died on a
    # quota error two cases from finishing and threw away eight hints that had
    # already been paid for -- on a free tier, that is most of a day.
    results = []
    if path.exists():
        results = json.loads(path.read_text())
        if len(results) >= len(cases):
            print(f"{path.name} is already complete. Delete it to regenerate.")
            return
        print(f"resuming: {len(results)} of {len(cases)} already done")

    for i, case in enumerate(cases, 1):
        if i <= len(results):
            continue
        question = await _load_question(case["question_id"], case["stem_starts_with"])
        produce = _agent if arm == "agent" else _direct
        generated = await produce(question, case["selected"])
        generated.case = case
        # The lexical guardrail runs in production; record what it would have
        # done so the judge's spoiler verdict can be compared against it.
        generated.lexical_guardrail_blocked = int(
            guardrail.check(generated.hint, generated.correct, generated.stem).leaked
        )
        results.append(asdict(generated))
        path.write_text(json.dumps(results, indent=2))
        print(
            f"  {i}/{len(cases)}  {generated.steps} steps  "
            f"{generated.latency_ms:>6.0f}ms  {generated.tools_used}"
        )

    print(f"\nwrote {len(results)} hints to {path.name}")


async def _judge_one(item: dict) -> dict:
    material = "\n\n".join(item["material"]) or "(none retrieved)"
    prompt = (
        f"QUESTION\n{item['stem']}\n\n"
        f"WHAT THE LEARNER CHOSE (incorrect)\n{item['chosen']}\n\n"
        f"THE CORRECT ANSWER (the tutor did not know this)\n{item['correct']}\n\n"
        f"REFERENCE MATERIAL AVAILABLE TO THE TUTOR\n{material}\n\n"
        f"THE HINT TO GRADE\n{item['hint']}"
    )
    completion = await llm.complete(
        system=JUDGE_SYSTEM,
        user=prompt,
        max_output_tokens=600,
        json_schema=JUDGE_SCHEMA,
        model=JUDGE_MODEL,
        rate_limit_wait=RATE_LIMIT_WAIT,
    )
    return json.loads(completion.text)


async def judge() -> None:
    runs = sorted(CACHE.glob("*.json"))
    if not runs:
        raise SystemExit("no generated runs. Run `generate` first.")

    header = (
        f"{'run':<34}{'grounded':>10}{'targeted':>10}"
        f"{'no spoiler':>12}{'actionable':>12}{'steps':>7}"
    )
    print(header)
    print("-" * len(header))

    for path in runs:
        items = json.loads(path.read_text())
        scores = {c: 0 for c in CRITERIA}
        verdicts = []
        for item in items:
            verdict = await _judge_one(item)
            verdicts.append(verdict)
            for criterion in CRITERIA:
                scores[criterion] += int(verdict[criterion])
        n = len(items)
        steps = sum(i["steps"] for i in items) / n
        print(
            f"{path.stem:<34}"
            + "".join(
                f"{scores[c] / n:>10.0%}" if c != "no_spoiler" else f"{scores[c] / n:>12.0%}"
                for c in CRITERIA[:3]
            )
            + f"{scores['actionable'] / n:>12.0%}{steps:>7.1f}"
        )
        path.with_suffix(".judged.json").write_text(
            json.dumps(
                [{"hint": i["hint"], **v} for i, v in zip(items, verdicts, strict=True)],
                indent=2,
            )
        )

    print("\nJudgements written alongside each run. Read them: an LLM grading an")
    print("LLM is a proxy, and one nobody has disagreed with is not validated.")


def show() -> None:
    """Print hints and verdicts for a human to check the judge against."""
    for path in sorted(CACHE.glob("*.judged.json")):
        print("=" * 78)
        print(path.stem)
        print("=" * 78)
        for entry in json.loads(path.read_text()):
            flags = "".join(("+" if entry[c] else "!") + c[0].upper() for c in CRITERIA)
            print(f"\n[{flags}] {entry['hint']}")
            for criterion in CRITERIA:
                if not entry[criterion]:
                    print(f"      {criterion}: {entry[criterion + '_reason']}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "judge", "show"])
    parser.add_argument("--arm", default="agent", choices=["agent", "direct"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.command == "show":
        show()
        return

    await db.pool.open()
    await db.pool.wait(timeout=10)
    try:
        if args.command == "generate":
            await generate(args.arm, args.model)
        else:
            await judge()
    finally:
        await db.pool.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show()
    else:
        asyncio.run(main())
