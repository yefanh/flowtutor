"""The tools the tutor can call.

WHAT A TOOL IS
    A capability described well enough that a model can decide, on its own,
    when to reach for it. Two halves: a schema the model reads, and a function
    it never sees.

    The description is the interface. A model picks tools by reading them, so a
    vague description produces a tool called at the wrong moments -- which
    looks like a reasoning failure and is really a writing one.

WHAT THE MODEL IS NOT ALLOWED TO DECIDE
    Every tool below is bound to a learner and a question before it is offered.
    `user_id` is not a parameter, so there is no way for the model to ask for
    somebody else's history, and no prompt injection can make it. The schema
    exposes what the model should choose -- the search phrasing, the note to
    keep -- and nothing else.

    This matters more than it looks. The model's inputs include retrieved text
    and, from the memory tool, text an earlier model wrote. Treating any of it
    as instructions would be a mistake; the tool boundary is where that is
    enforced structurally instead of hopefully.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import db
from adaptive import model as mastery_model
from tutor import llm, retrieval

MAX_REMEMBERED_NOTES = 5
"""How many observations to keep per concept.

A memory that only grows becomes a wall of text nobody reads, including the
model -- and the older the note, the more likely the learner has moved past it.
"""


@dataclass
class Toolbox:
    """Tools bound to one learner working on one question."""

    specs: list[llm.ToolSpec]
    handlers: dict[str, Callable[[dict], Awaitable[dict]]]

    async def run(self, name: str, arguments: dict) -> dict:
        handler = self.handlers.get(name)
        if handler is None:
            # Returned rather than raised: a model that hallucinates a tool
            # name should be told so and given another turn, not crash the
            # request.
            return {"error": f"no such tool: {name}"}
        return await handler(arguments)


def build(user_id: int, question: dict) -> Toolbox:
    concept_id = question["concept_id"]
    question_id = question["id"]

    async def search_material(args: dict) -> dict:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "query was empty"}
        chunks = await retrieval.search(
            query, concept_id=concept_id, exclude_question_id=question_id
        )
        return {"results": [{"source": c.citation, "content": c.content} for c in chunks]}

    async def recall_learner(_args: dict) -> dict:
        """Facts, not interpretation.

        Everything here is derived from the attempt log and the mastery table,
        so it costs a query rather than a model call and cannot drift out of
        date.
        """
        score = await db.query_one(
            """
            SELECT COALESCE(m.score, %s) AS score, COALESCE(m.attempts, 0) AS attempts
            FROM concepts c
            LEFT JOIN mastery m ON m.concept_id = c.id AND m.user_id = %s
            WHERE c.id = %s
            """,
            (mastery_model.DEFAULT_MASTERY, user_id, concept_id),
        )
        recent = await db.query_all(
            """
            SELECT q.stem, q.options, a.selected, a.created_at
            FROM attempts a
            JOIN questions q ON q.id = a.question_id
            WHERE a.user_id = %s AND q.concept_id = %s
              AND a.is_correct = FALSE AND q.id <> %s
            ORDER BY a.created_at DESC
            LIMIT 4
            """,
            (user_id, concept_id, question_id),
        )
        notes = await _read_notes(user_id, concept_id)
        return {
            "mastery": round(float(score["score"]), 2),
            "questions_answered_in_this_concept": score["attempts"],
            "recent_wrong_answers": [
                {"question": r["stem"], "they_chose": r["options"][r["selected"]]} for r in recent
            ],
            "notes_from_earlier_sessions": notes,
        }

    async def remember(args: dict) -> dict:
        note = str(args.get("note", "")).strip()
        if not note:
            return {"error": "note was empty"}
        notes = await _read_notes(user_id, concept_id)
        if note in notes:
            return {"stored": False, "reason": "already recorded"}
        notes = [note, *notes][:MAX_REMEMBERED_NOTES]
        await db.execute(
            """
            INSERT INTO learner_memory (user_id, key, value)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, key)
            DO UPDATE SET value = EXCLUDED.value, updated_at = now()
            """,
            (user_id, _notes_key(concept_id), json.dumps({"notes": notes})),
        )
        return {"stored": True, "total_notes": len(notes)}

    specs = [
        llm.ToolSpec(
            name="search_material",
            description=(
                "Search the lesson material and worked explanations for this "
                "concept. Use it whenever you need to ground a hint in what the "
                "learner has actually been taught, or when a first search came "
                "back unhelpful and a different phrasing might do better. "
                "Returns passages with the lesson step they came from."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to look for, phrased as the idea you are after "
                            "rather than as keywords."
                        ),
                    }
                },
                "required": ["query"],
            },
        ),
        llm.ToolSpec(
            name="recall_learner",
            description=(
                "Look up what is known about this learner on this concept: their "
                "current mastery, the questions they recently got wrong and what "
                "they picked, and any notes kept from earlier sessions. Use it "
                "when their mistake might be part of a pattern rather than a "
                "one-off, so the hint can address the pattern."
            ),
            parameters={"type": "object", "properties": {}},
        ),
        llm.ToolSpec(
            name="remember",
            description=(
                "Record one durable observation about how this learner thinks, "
                "for future sessions. Only for patterns worth carrying forward, "
                "such as a recurring confusion between two ideas. Do not record "
                "single mistakes, scores, or anything already visible in their "
                "answer history."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": (
                            "One sentence, about the learner's thinking rather "
                            "than about this question."
                        ),
                    }
                },
                "required": ["note"],
            },
        ),
    ]

    return Toolbox(
        specs=specs,
        handlers={
            "search_material": search_material,
            "recall_learner": recall_learner,
            "remember": remember,
        },
    )


async def standing_summary(user_id: int, concept_id: int) -> str:
    """One line on how a learner is doing, cheap enough to always include.

    This is the counterpart to `recall_learner`: the summary says whether there
    is anything to look into, the tool says what it is. Splitting them that way
    is what makes calling the tool a decision the model can actually make.
    """
    row = await db.query_one(
        """
        SELECT COALESCE(m.score, %s) AS score,
               (SELECT count(*) FROM attempts a
                 JOIN questions q ON q.id = a.question_id
                WHERE a.user_id = %s AND q.concept_id = %s) AS tries,
               (SELECT count(*) FROM attempts a
                 JOIN questions q ON q.id = a.question_id
                WHERE a.user_id = %s AND q.concept_id = %s
                  AND a.is_correct = FALSE) AS wrong
        FROM concepts c
        LEFT JOIN mastery m ON m.concept_id = c.id AND m.user_id = %s
        WHERE c.id = %s
        """,
        (
            mastery_model.DEFAULT_MASTERY,
            user_id,
            concept_id,
            user_id,
            concept_id,
            user_id,
            concept_id,
        ),
    )
    if row is None:
        return ""

    score = float(row["score"])
    if row["tries"] == 0:
        return "First question on this concept; nothing known about them yet."

    notes = await _read_notes(user_id, concept_id)
    line = (
        f"Mastery {score:.2f} out of 1. "
        f"{row['wrong']} of {row['tries']} questions on this concept answered "
        "wrong so far."
    )
    if notes:
        line += f" {len(notes)} note(s) kept from earlier sessions."
    return line


def _notes_key(concept_id: int) -> str:
    return f"concept:{concept_id}:notes"


async def _read_notes(user_id: int, concept_id: int) -> list[str]:
    row = await db.query_one(
        "SELECT value FROM learner_memory WHERE user_id = %s AND key = %s",
        (user_id, _notes_key(concept_id)),
    )
    if row is None:
        return []
    return list(row["value"].get("notes", []))
