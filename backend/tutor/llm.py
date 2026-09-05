"""Talking to a language model.

ONE THIN INTERFACE, SEVERAL POSSIBLE PROVIDERS
    Everything above this module asks for `complete(system, user)` and gets
    text back. Who answers is a configuration value.

    That was not speculative flexibility, and it has now paid for itself. Free
    tiers turned out to be the binding constraint on this project three times
    over: which model could be used at all, whether the agent could plan across
    tools, and whether the evaluation could finish inside a day. Gemini Pro
    went behind billing in May 2026; the newest Flash returned 503 the first
    time it was tried; Flash-Lite allows fifteen requests a minute.

    The default is now a model running on this machine. Switching back to a
    hosted one is an env var, which is exactly what this layer was for.

MODEL FALLBACK
    Same reason, one level down. The newest model is the one most likely to be
    rate limited or overloaded for free-tier traffic, so a 503 or 429 falls
    through to the next model rather than failing the request.

    Both failures were observed rather than anticipated. gemini-3.8-flash
    returned 503 "experiencing high demand" on the very first probe. Then
    gemini-3.5-flash, which had been working, started returning 429 with
    "limit: 20, GenerateRequestsPerDayPerProjectPerModel-FreeTier" -- twenty
    requests a day, exhausted by an afternoon of testing.

    Hence the default is a Flash-Lite model: on the free tier the quota is the
    binding constraint, not the capability. Being able to iterate is worth more
    than a marginally better sentence, and the fallbacks cover the rest.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
FALLBACK_MODELS: list[str] = []

# Only used when LLM_PROVIDER=google. Kept so switching back is an env var.
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite")
GOOGLE_FALLBACKS = ["gemini-flash-latest", "gemini-3.5-flash"]

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Low but not zero. Hints should be stable enough that evaluating one tells you
# something about the next, without being word-for-word identical every time.
TEMPERATURE = 0.3

THINKING_BUDGET = int(os.getenv("LLM_THINKING_BUDGET", "0"))
"""Tokens the model may spend reasoning before it answers.

Gemini 3.x thinks by default, and measured on a trivial prompt it spent 301
thinking tokens to produce 7 -- 1757ms against 685ms with thinking off. For a
task this shaped (read five passages, write three sentences, do not state the
answer) that is a lot of latency for reasoning nobody reads.

Set to 0 by default and treated as a tunable rather than a decision: if hint
quality or answer leakage turns out to need it, this goes up and the cost is
visible in the eval. Guessing either way would be guessing.
"""


class LLMError(RuntimeError):
    """The model could not be reached, or every fallback was exhausted."""


@dataclass(frozen=True)
class ToolCall:
    """The model asking for a tool to be run. Arguments are already parsed."""

    name: str
    arguments: dict


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tool_calls: tuple[ToolCall, ...] = ()
    raw: object | None = None
    """The provider's own representation of this turn, kept opaque.

    Gemini rejects a replayed function call that was rebuilt from its parsed
    fields -- the call carries a `thought_signature` that has to come back
    byte-for-byte, and a reconstruction drops it:

        400 INVALID_ARGUMENT: Function call is missing a thought_signature in
        functionCall parts.

    So the original object is carried through and echoed back verbatim. Nothing
    above this module reads it; the agent copies it from a Completion into a
    Turn and never looks inside.
    """

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the model sees it.

    The description is not documentation, it is the interface. The model
    chooses tools by reading it, so a vague description produces a tool that is
    called at the wrong times -- which looks like a reasoning failure and is
    actually a writing one.
    """

    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class Turn:
    """One entry in the conversation the agent is building up.

    `role` is "user", "model", or "tool". A tool turn carries the result of a
    call the model asked for on the previous turn.
    """

    role: str
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_name: str | None = None
    tool_result: dict | None = None
    raw: object | None = None


async def complete(
    system: str,
    user: str | None = None,
    max_output_tokens: int = 400,
    turns: list[Turn] | None = None,
    tools: list[ToolSpec] | None = None,
    json_schema: dict | None = None,
    model: str | None = None,
    rate_limit_wait: float = 0.0,
    temperature: float | None = None,
) -> Completion:
    """One model call.

    Either `user` (a single prompt) or `turns` (a running conversation, which
    is what the agent loop passes).

    `json_schema` constrains the reply to that shape. Used by the evaluation
    judge: a grader that answers in prose has to be parsed with a regex, and a
    grader whose output occasionally fails to parse silently drops the cases it
    was least sure about -- exactly the ones worth reading.

    `model` overrides the configured default, for A/B runs.

    `rate_limit_wait` is the longest this call may sleep and retry when the
    provider says "too fast". Zero in the request path -- a learner waiting for
    a hint should get a fallback model or an error, not a 35-second pause. Set
    high for batch jobs, where the alternative is a run that dies a third of
    the way through.
    """
    if turns is None:
        turns = [Turn(role="user", text=user or "")]

    if PROVIDER == "ollama":
        return await _ollama(
            system, turns, tools or [], max_output_tokens, json_schema, model, temperature
        )
    if PROVIDER == "google":
        return await _google(
            system,
            turns,
            tools or [],
            max_output_tokens,
            json_schema,
            model or GOOGLE_MODEL,
            rate_limit_wait,
        )
    raise LLMError(f"unknown LLM_PROVIDER: {PROVIDER!r}")


@lru_cache(maxsize=1)
def _ollama_client():
    from ollama import AsyncClient

    return AsyncClient(host=OLLAMA_HOST)


def _to_ollama_message(turn: "Turn") -> dict:
    if turn.role == "tool":
        return {
            "role": "tool",
            "tool_name": turn.tool_name or "",
            "content": json.dumps(turn.tool_result or {}),
        }
    if turn.tool_calls:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in turn.tool_calls
            ],
        }
    return {
        "role": "assistant" if turn.role == "model" else "user",
        "content": turn.text,
    }


async def _ollama(
    system: str,
    turns: list["Turn"],
    tools: list["ToolSpec"],
    max_output_tokens: int,
    json_schema: dict | None,
    model_override: str | None,
    temperature: float | None = None,
) -> Completion:
    """A model running on this machine.

    No key, no quota, no rate limit, no network. That is the whole reason it is
    here: free-tier limits had become the binding constraint on this project
    three separate times -- which model could be used at all, whether the agent
    could plan across tools, and whether the evaluation could finish in a day.

    A 9B model writes a weaker sentence than a frontier one. It also runs as
    many times as you like, which for an evaluation harness -- something that
    has to be re-runnable, and eventually has to run in CI -- matters more than
    the sentence.
    """
    client = _ollama_client()
    model = model_override or MODEL
    started = time.perf_counter()

    declarations = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]

    try:
        response = await client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                *[_to_ollama_message(turn) for turn in turns],
            ],
            tools=declarations or None,
            format=json_schema,
            # Qwen3 thinks by default, and its reasoning does not come back in
            # `content` -- it goes to a separate field. Left on, a request for
            # structured output returned an EMPTY string: the whole token
            # budget was spent thinking and there was nothing left to answer
            # with. The same knob that governs the hosted provider governs this
            # one, so the two behave alike.
            think=THINKING_BUDGET > 0,
            options={
                "temperature": TEMPERATURE if temperature is None else temperature,
                "num_predict": max_output_tokens,
            },
        )
    except Exception as exc:  # noqa: BLE001 -- provider SDKs raise their own types
        raise LLMError(
            f"{model}: {exc}. Is Ollama running? Try `brew services start ollama` "
            f"and `ollama pull {model}`."
        ) from exc

    message = response.message
    calls = tuple(
        ToolCall(
            name=call.function.name or "",
            arguments=dict(call.function.arguments or {}),
        )
        for call in (message.tool_calls or [])
    )
    return Completion(
        text=(message.content or "").strip(),
        model=model,
        input_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
        latency_ms=(time.perf_counter() - started) * 1000,
        tool_calls=calls,
    )


@lru_cache(maxsize=1)
def _client():
    """One client for the process.

    Constructing it per request cost seconds on the first call -- auth and
    transport setup that has no reason to repeat.
    """
    from google import genai

    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise LLMError(
            "GOOGLE_API_KEY is not set. Copy backend/.env.example to backend/.env "
            "and paste a key from https://aistudio.google.com"
        )
    return genai.Client(api_key=key)


def _to_google_content(turn: "Turn", types):
    """Translate one turn into the provider's wire format.

    Kept in one place so the rest of the codebase never sees a provider type.
    """
    if turn.role == "tool":
        return types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=turn.tool_name or "", response=turn.tool_result or {}
                )
            ],
        )

    if turn.tool_calls:
        # Replay the provider's own object when we have it. Rebuilding the call
        # from its parsed fields loses the thought signature Gemini requires.
        if turn.raw is not None:
            return turn.raw
        return types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(name=call.name, args=call.arguments)
                for call in turn.tool_calls
            ],
        )

    return types.Content(
        role="model" if turn.role == "model" else "user",
        parts=[types.Part.from_text(text=turn.text)],
    )


async def _google(
    system: str,
    turns: list["Turn"],
    tools: list["ToolSpec"],
    max_output_tokens: int,
    json_schema: dict | None = None,
    model_override: str | None = None,
    rate_limit_wait: float = 0.0,
) -> Completion:
    from google.genai import types

    client = _client()
    declarations = [
        types.FunctionDeclaration(
            name=tool.name, description=tool.description, parameters=tool.parameters
        )
        for tool in tools
    ]
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_output_tokens,
        temperature=TEMPERATURE,
        # The SDK will happily execute tools for us. It must not: the agent
        # loop is the thing being built here, and hiding it inside the client
        # library would hide exactly the mechanism worth understanding.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
        tools=[types.Tool(function_declarations=declarations)] if declarations else None,
        response_mime_type="application/json" if json_schema else None,
        response_schema=json_schema,
    )
    contents = [_to_google_content(turn, types) for turn in turns]

    # Google's own list, not the top-level MODEL, which now names a local one.
    candidates = [model_override] if model_override else [GOOGLE_MODEL, *GOOGLE_FALLBACKS]
    errors: list[str] = []
    for model in candidates:
        started = time.perf_counter()
        try:
            response = await client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:  # noqa: BLE001 -- provider SDKs raise their own types
            message = str(exc).split("\n")[0]

            # The free tier limits requests per MINUTE (15 on Flash-Lite), and
            # the error carries the exact wait the server wants. Honouring it
            # is the difference between a batch job that finishes and one that
            # dies a third of the way through.
            delay = _suggested_retry_delay(str(exc))
            if delay and delay <= rate_limit_wait:
                await asyncio.sleep(delay + 0.5)
                try:
                    response = await client.aio.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    message = str(retry_exc).split("\n")[0]
                    errors.append(f"{model}: {message}")
                    continue
            else:
                errors.append(f"{model}: {message}")
                # Overload and rate limiting are the failures a fallback
                # exists for. Anything else (bad key, bad request) fails the
                # same way on every model, so there is no point burning the
                # fallbacks on it.
                if not any(code in message for code in ("503", "429", "500", "UNAVAILABLE")):
                    break
                continue

        calls = tuple(
            ToolCall(name=c.name or "", arguments=dict(c.args or {}))
            for c in (response.function_calls or [])
        )
        usage = response.usage_metadata
        candidate = (response.candidates or [None])[0]
        return Completion(
            text=(response.text or "").strip() if not calls else "",
            model=model,
            input_tokens=usage.prompt_token_count or 0,
            output_tokens=usage.candidates_token_count or 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            tool_calls=calls,
            raw=getattr(candidate, "content", None) if calls else None,
        )

    raise LLMError("; ".join(errors) or "no model responded")


def _suggested_retry_delay(message: str) -> float | None:
    """The wait the provider asked for, if it said."""
    match = re.search(r"'retryDelay': '(\d+)s'", message)
    return float(match.group(1)) if match else None
