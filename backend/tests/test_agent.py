"""The agent loop and its tools.

The model is scripted throughout. What is being tested is the loop -- that it
stops, that it records what it did, that a tool cannot reach past the learner
it was built for -- none of which is the model's behaviour.
"""

import db
from tutor import agent, hints, llm, tools

CACHING = 1


async def _question(difficulty: int = 2) -> dict:
    return await db.query_one(
        """
        SELECT q.id, q.concept_id, c.name AS concept_name,
               q.stem, q.options, q.answer, q.difficulty, q.explanation
        FROM questions q
        JOIN concepts c ON c.id = q.concept_id
        WHERE q.concept_id = %s AND q.difficulty = %s
        LIMIT 1
        """,
        (CACHING, difficulty),
    )


async def _reset(user_id: int) -> None:
    await db.execute("DELETE FROM hints WHERE user_id = %s", (user_id,))
    await db.execute("DELETE FROM attempts WHERE user_id = %s", (user_id,))
    await db.execute("DELETE FROM mastery WHERE user_id = %s", (user_id,))
    await db.execute("DELETE FROM learner_memory WHERE user_id = %s", (user_id,))


# ------------------------------------------------------------------- the loop


async def test_an_immediate_answer_takes_no_steps(client, fake_llm):
    """A tool-using agent is still allowed to just answer."""
    question = await _question()
    fake_llm["replies"].append("See lesson step 4.")

    run = await agent.run(system="s", task="t", toolbox=tools.build(9401, question))
    assert run.text == "See lesson step 4."
    assert run.steps == []
    assert not run.hit_step_limit


async def test_a_tool_call_is_executed_and_fed_back(client, fake_llm):
    question = await _question()
    fake_llm["replies"].extend(
        [
            [llm.ToolCall(name="search_material", arguments={"query": "stale data"})],
            "See lesson step 4.",
        ]
    )

    run = await agent.run(system="s", task="t", toolbox=tools.build(9402, question))

    assert len(run.steps) == 1
    assert run.steps[0].tool == "search_material"
    assert run.sources
    # The tool result went back to the model on the next turn.
    final_turns = fake_llm["sent"][-1]["turns"]
    assert any(t.role == "tool" for t in final_turns)


async def test_the_step_limit_stops_a_model_that_never_answers(client, fake_llm):
    """A termination guarantee, not a tuning knob."""
    question = await _question()
    fake_llm["replies"].extend(
        [[llm.ToolCall(name="search_material", arguments={"query": "cache"})]] * 20
    )

    run = await agent.run(system="s", task="t", toolbox=tools.build(9403, question), max_steps=3)

    assert run.hit_step_limit
    assert len(run.steps) <= 3


async def test_tools_are_withheld_on_the_final_pass(client, fake_llm):
    """Asking a model not to call tools while still offering them is a request.
    Taking them away makes an answer the only thing it can produce."""
    question = await _question()
    fake_llm["replies"].extend(
        [[llm.ToolCall(name="search_material", arguments={"query": "cache"})]] * 20
    )

    await agent.run(system="s", task="t", toolbox=tools.build(9404, question), max_steps=2)

    assert fake_llm["sent"][-1]["tools"] == []
    assert fake_llm["sent"][0]["tools"]


async def test_an_unknown_tool_is_reported_not_raised(client, fake_llm):
    """A hallucinated tool name should cost a turn, not the request."""
    question = await _question()
    fake_llm["replies"].extend(
        [[llm.ToolCall(name="look_up_the_answer", arguments={})], "See lesson step 4."]
    )

    run = await agent.run(system="s", task="t", toolbox=tools.build(9405, question))

    assert run.text == "See lesson step 4."
    assert "error" in run.steps[0].result


async def test_the_trace_records_what_happened(client, fake_llm):
    question = await _question()
    fake_llm["replies"].extend(
        [
            [llm.ToolCall(name="recall_learner", arguments={})],
            [llm.ToolCall(name="search_material", arguments={"query": "stale"})],
            "See lesson step 4.",
        ]
    )

    run = await agent.run(system="s", task="t", toolbox=tools.build(9406, question))
    trace = run.trace()

    assert [entry["tool"] for entry in trace] == ["recall_learner", "search_material"]
    # Passages are already in the database; the trace says WHICH ones came back.
    assert "sources" in trace[1]["result"]
    assert all("duration_ms" in entry for entry in trace)


async def test_the_trace_is_stored_with_the_hint(client, fake_llm):
    """An agent is several decisions deep. Without the trace, a bad search and
    a bad write-up look identical from the outside."""
    user = 9407
    await _reset(user)
    question = await _question()
    wrong = (question["answer"] + 1) % len(question["options"])

    fake_llm["replies"].extend(
        [
            [llm.ToolCall(name="search_material", arguments={"query": "passive"})],
            "See lesson step 6.",
        ]
    )
    await hints.generate(user, question, wrong)

    row = await db.query_one(
        "SELECT steps, trace FROM hints WHERE user_id = %s ORDER BY id DESC LIMIT 1",
        (user,),
    )
    assert row["steps"] == 1
    assert row["trace"][0]["tool"] == "search_material"


# -------------------------------------------------------------------- tools


async def test_recall_reports_facts_from_the_learners_history(client, fake_llm):
    user = 9408
    await _reset(user)
    question = await _question()

    other = await db.query_one(
        "SELECT id, options, answer FROM questions WHERE concept_id = %s AND id <> %s LIMIT 1",
        (CACHING, question["id"]),
    )
    wrong = (other["answer"] + 1) % len(other["options"])
    await db.execute(
        """
        INSERT INTO attempts (user_id, question_id, selected, is_correct)
        VALUES (%s, %s, %s, FALSE)
        """,
        (user, other["id"], wrong),
    )

    box = tools.build(user, question)
    result = await box.run("recall_learner", {})

    assert result["recent_wrong_answers"]
    assert result["recent_wrong_answers"][0]["they_chose"] == other["options"][wrong]
    # The question they are on right now is excluded -- it is in front of them.
    assert all(r["question"] != question["stem"] for r in result["recent_wrong_answers"])


async def test_remember_writes_a_note_that_recall_reads_back(client, fake_llm):
    user = 9409
    await _reset(user)
    question = await _question()
    box = tools.build(user, question)

    assert (await box.run("remember", {"note": "Confuses caching with durability."}))["stored"]
    result = await box.run("recall_learner", {})
    assert "Confuses caching with durability." in result["notes_from_earlier_sessions"]


async def test_remember_does_not_duplicate_a_note(client, fake_llm):
    user = 9410
    await _reset(user)
    box = tools.build(user, await _question())

    await box.run("remember", {"note": "Reaches for capacity fixes."})
    second = await box.run("remember", {"note": "Reaches for capacity fixes."})
    assert second["stored"] is False


async def test_notes_are_capped(client, fake_llm):
    """A memory that only grows becomes a wall of text nobody reads."""
    user = 9411
    await _reset(user)
    box = tools.build(user, await _question())

    for i in range(tools.MAX_REMEMBERED_NOTES + 3):
        await box.run("remember", {"note": f"Observation number {i}."})

    result = await box.run("recall_learner", {})
    assert len(result["notes_from_earlier_sessions"]) == tools.MAX_REMEMBERED_NOTES
    # Newest first, so the cap drops the stalest observations.
    assert "number 7" in result["notes_from_earlier_sessions"][0]


async def test_memory_is_per_learner(client, fake_llm):
    user_a, user_b = 9412, 9413
    await _reset(user_a)
    await _reset(user_b)
    question = await _question()

    await tools.build(user_a, question).run("remember", {"note": "A's pattern."})
    result = await tools.build(user_b, question).run("recall_learner", {})

    assert result["notes_from_earlier_sessions"] == []


async def test_no_tool_takes_a_user_id(client, fake_llm):
    """The model cannot ask for somebody else's history, because the schema
    gives it no way to name one. Not a rule it is asked to follow."""
    box = tools.build(9414, await _question())
    for spec in box.specs:
        assert "user_id" not in spec.parameters.get("properties", {})
        assert "user" not in str(spec.parameters).lower()
