"""The agent loop.

WHAT AN AGENT IS
    Not a model. A loop around one.

        plan  -> the model looks at the situation and picks a tool
        act   -> our code runs that tool
        observe -> the result goes back into the conversation
        repeat until the model stops asking for tools and just answers

    The model supplies the judgement; the loop supplies the turns. Everything
    that makes this agentic lives in the twenty lines of `run` below -- the
    model is the same one that answered a single prompt in Phase 2.

WHY A LOOP AT ALL, HERE
    Phase 2's hint pipeline was retrieve, generate, verify -- a fixed sequence
    with nothing to decide. Wrapping a loop around that would have been
    decoration: a model given one option is not choosing.

    It earns its place once the choices are real. Was one search enough, or
    should it try different words? Is this mistake a one-off, or does this
    learner keep making it? Is there something here worth remembering for next
    time? Those are decisions that depend on what came back, which is exactly
    what a fixed pipeline cannot express.

WHETHER IT IS EARNING ITS KEEP (measured, and the answer is "depends")
    An agent with tools it never calls is a slower pipeline. That is what this
    was, at first: given a learner sitting at 0.12 mastery with four straight
    wrong answers, the tutor searched once and answered, never once looking at
    who it was talking to.

    Two things were tried. Telling it up front how the learner is doing -- so
    that "is there a pattern here" stopped being a question only the tool could
    answer -- changed nothing. Changing the model did:

        model                    steps   tools called
        gemini-3.1-flash-lite        1   search
        gemini-flash-latest          1   search
        gemini-3.5-flash             2   search, recall_learner

    So multi-tool planning here is a capability question, and the model with a
    workable free quota does not have it. The loop is correct either way -- the
    tests pin the mechanics -- but on Flash-Lite it currently behaves like the
    fixed pipeline it replaced, at a little more latency.

    That is a tradeoff to know about, not a bug to hide: switching is one env
    var, and Phase 4's evaluation is what will say whether hints that used
    recall are actually better than hints that did not. Until then, "the agent
    helps" is a hypothesis.

WHY IT IS HAND-WRITTEN
    The provider SDK will run this loop for us, and so will several frameworks.
    Both hide the one mechanism worth understanding. The loop is small; owning
    it means the step limit, the trace, and the failure handling are visible
    and adjustable rather than configuration options in someone else's library.

    Provider function calling IS used for the tool protocol itself -- getting
    structured calls back beats parsing them out of prose. The loop is ours;
    the wire format is theirs.
"""

import time
from dataclasses import dataclass, field

from tutor import llm, tools

MAX_STEPS = 5
"""Hard ceiling on tool calls in one request.

Not a performance tuning knob -- a termination guarantee. A model that keeps
searching, or gets into a loop of near-identical calls, would otherwise run
until something else times out. When the limit is hit the agent is asked for
its best answer with what it has, which is a worse hint than it might have
written and infinitely better than none.
"""


@dataclass
class Step:
    """One tool call, as it happened."""

    tool: str
    arguments: dict
    result: dict
    duration_ms: float

    def summarised(self) -> dict:
        """A form small enough to store on every hint.

        Retrieved passages are hundreds of words each and are already in the
        database; the trace needs to say WHICH ones came back, not repeat them.
        """
        result = self.result
        if "results" in result:
            summary = {"sources": [r["source"] for r in result["results"]]}
        else:
            summary = {k: v for k, v in result.items() if k != "notes_from_earlier_sessions"}
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "result": summary,
            "duration_ms": round(self.duration_ms),
        }


@dataclass
class AgentRun:
    text: str
    steps: list[Step] = field(default_factory=list)
    model: str | None = None
    latency_ms: float = 0.0
    hit_step_limit: bool = False

    @property
    def sources(self) -> list[str]:
        """Every passage the agent actually looked at, in order, deduplicated."""
        seen: list[str] = []
        for step in self.steps:
            for result in step.result.get("results", []):
                if result["source"] not in seen:
                    seen.append(result["source"])
        return seen

    def trace(self) -> list[dict]:
        return [step.summarised() for step in self.steps]


async def run(
    system: str,
    task: str,
    toolbox: tools.Toolbox,
    max_steps: int = MAX_STEPS,
    max_output_tokens: int = 320,
    rate_limit_wait: float = 0.0,
) -> AgentRun:
    """Plan, act, observe, repeat.

    Returns as soon as the model answers with text instead of a tool call.
    """
    turns = [llm.Turn(role="user", text=task)]
    result = AgentRun(text="")
    started = time.perf_counter()
    already_called: set[tuple[str, str]] = set()

    for step_number in range(max_steps + 1):
        last_step = step_number == max_steps

        completion = await llm.complete(
            system=system,
            turns=turns,
            # On the final pass the tools are withheld entirely. Asking a model
            # not to call tools while still offering them is a request; taking
            # them away makes an answer the only thing it can produce.
            tools=None if last_step else toolbox.specs,
            max_output_tokens=max_output_tokens,
            rate_limit_wait=rate_limit_wait,
        )
        result.model = completion.model

        if not completion.wants_tool:
            result.text = completion.text
            break

        if last_step:
            result.hit_step_limit = True
            break

        turns.append(
            llm.Turn(
                role="model",
                tool_calls=completion.tool_calls,
                # Opaque provider state, replayed unread.
                raw=completion.raw,
            )
        )

        # Calls arrive in a batch and are run in order. Sequential on purpose:
        # `remember` writes, and a model that both reads and writes memory in
        # one turn should see them happen in the order it asked for.
        for call in completion.tool_calls:
            call_started = time.perf_counter()
            signature = (call.name, repr(sorted(call.arguments.items())))

            if signature in already_called:
                # Observed: a model searched for the same phrase five times in
                # a row and burned the whole step budget. The limit stopped it,
                # but stopping is not the same as helping -- repeating a call
                # is usually a model that did not notice it had the answer
                # already, so say so instead of running the query again.
                output = {
                    "error": (
                        "You already ran this exact call and have the result "
                        "above. Use it, or try something different."
                    )
                }
            else:
                already_called.add(signature)
                output = await toolbox.run(call.name, call.arguments)

            duration = (time.perf_counter() - call_started) * 1000

            result.steps.append(
                Step(
                    tool=call.name,
                    arguments=call.arguments,
                    result=output,
                    duration_ms=duration,
                )
            )
            turns.append(llm.Turn(role="tool", tool_name=call.name, tool_result=output))
    else:
        result.hit_step_limit = True

    result.latency_ms = (time.perf_counter() - started) * 1000
    return result
