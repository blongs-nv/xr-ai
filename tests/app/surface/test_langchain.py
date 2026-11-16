# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optional LangChain/LangGraph adapters (``xr_ai_kit.langchain``), and samples 08 and 09 on fake models.

Skipped unless ``langchain-core`` and ``langgraph`` (the kit's ``langchain`` extra) are installed; see
``app-sdk/xr-ai-kit/xr_ai_kit/langchain/README.md``.
"""

from __future__ import annotations

import asyncio
from operator import add
from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")
pytest.importorskip("xr_ai_kit.langchain")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool as lc_tool  # noqa: E402
from langgraph.config import get_stream_writer  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402
from xr_ai_app import ContextLog, ContextValue, TextSink, tool  # noqa: E402
from xr_ai_app._base.participant import CURRENT_TURN  # noqa: E402
from xr_ai_app.testing import FakeModels, ToolCall, solid_frame  # noqa: E402
from xr_ai_kit import Conversation  # noqa: E402
from xr_ai_kit.delegate import Job, JobUpdate  # noqa: E402
from xr_ai_kit.langchain import (  # noqa: E402
    GraphBackend,
    chat_model,
    from_langchain,
    langchain_tools,
    messages,
    stream_text,
    to_chat_messages,
    to_langchain_messages,
)
from xr_ai_models import ChatResponse, ChatStreamEvent  # noqa: E402
from xr_ai_models import ToolCall as ModelToolCall  # noqa: E402


async def until(s, condition, *, timeout_s: float = 3.0) -> None:
    """Settle repeatedly until ``condition()`` holds (background work that waits on time, not on models)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        assert loop.time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)
        await s.settle()


@lc_tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


class _Recording:
    """A role whose ``as_llm()`` records what it is sent and answers from a script of ``ChatResponse``."""

    name = "recording"

    def __init__(self, *responses: ChatResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def as_llm(self) -> _Recording:
        return self

    async def chat(self, messages, **options) -> ChatResponse:
        self.calls.append({"messages": list(messages), **options})
        return self.responses.pop(0)

    async def stream_events(self, messages, **options):
        response = await self.chat(messages, **options)
        if response.reasoning:
            yield ChatStreamEvent(kind="reasoning", reasoning=response.reasoning)
        for word in response.content.split(" ") if response.content else ():
            yield ChatStreamEvent(kind="text", text=word + " ")
        for call in response.tool_calls or ():
            yield ChatStreamEvent(kind="tool_call", tool_call=call)
        yield ChatStreamEvent(kind="finish", finish_reason=response.finish_reason)


def _answer(text: str, *calls: ModelToolCall, reasoning: str | None = None) -> ChatResponse:
    return ChatResponse(text, reasoning, list(calls) or None, "tool_calls" if calls else "stop", {})


# ---------------------------------------------------------------------- chat_model


async def test_chat_model_invoke_and_stream_on_a_fake_role() -> None:
    models = FakeModels({"r": ["Hello there.", "One two three."]})
    llm = chat_model(models.role("r"))
    reply = await llm.ainvoke([SystemMessage("Be brief."), SystemMessage("More rules."), HumanMessage("hi")])
    assert isinstance(reply, AIMessage) and reply.content == "Hello there." and reply.tool_calls == []
    (call,) = models.calls("r")
    assert call.system == "Be brief.\n\nMore rules." and call.prompt == "hi"

    chunks = [chunk async for chunk in llm.astream("count")]
    assert "".join(c.content for c in chunks) == "One two three."
    assert len([c for c in chunks if c.content]) == 3  # streamed word by word, not as one block


async def test_chat_model_sync_calls_raise_clearly() -> None:
    llm = chat_model(FakeModels({"r": []}).role("r"))
    with pytest.raises(NotImplementedError, match="async"):
        await asyncio.to_thread(llm.invoke, "hi")


async def test_chat_model_tool_call_round_trip_on_a_fake_role() -> None:
    models = FakeModels({"r": [ToolCall("multiply", {"a": 6, "b": 7}), "It is 42."]})
    llm = chat_model(models.role("r")).bind_tools([multiply])
    first = await llm.ainvoke([HumanMessage("what is 6 times 7?")])
    assert first.tool_calls == [{"name": "multiply", "args": {"a": 6, "b": 7}, "id": "call_0", "type": "tool_call"}]
    result = await multiply.ainvoke(first.tool_calls[0])
    assert isinstance(result, ToolMessage) and result.content == "42"
    second = await llm.ainvoke([HumanMessage("what is 6 times 7?"), first, result])
    assert second.content == "It is 42."
    prompt = models.calls("r")[1].prompt
    assert 'You called tool multiply with {"a": 6, "b": 7}' in prompt and "Tool multiply returned: 42" in prompt


async def test_bind_tools_and_message_conversion() -> None:
    call = ModelToolCall("c1", "multiply", '{"a": 2, "b": 3}')
    role = _Recording(_answer("", call, reasoning="thinking..."), _answer("Six."))
    llm = chat_model(role, max_tokens=64, enable_thinking=False).bind_tools([multiply], tool_choice="auto")

    first = await llm.ainvoke("2 times 3?")
    assert first.tool_calls[0]["args"] == {"a": 2, "b": 3} and first.tool_calls[0]["id"] == "c1"
    assert first.additional_kwargs["reasoning_content"] == "thinking..." and first.content == ""
    sent = role.calls[0]
    (definition,) = sent["tools"]
    assert definition.name == "multiply" and definition.description == "Multiply two integers."
    assert set(definition.parameters["properties"]) == {"a", "b"}
    assert sent["max_tokens"] == 64 and sent["enable_thinking"] is False

    chunks = [c async for c in llm.astream([HumanMessage("2 times 3?"), first, ToolMessage("6", tool_call_id="c1")])]
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    assert merged.content == "Six. " and merged.response_metadata.get("finish_reason") == "stop"
    transcript = role.calls[1]["messages"]
    assert [m.role for m in transcript] == ["user", "assistant", "tool"]
    assert transcript[1].tool_calls[0].name == "multiply" and transcript[2].tool_call_id == "c1"

    back = to_langchain_messages(to_chat_messages([HumanMessage("x"), first, ToolMessage("6", tool_call_id="c1")]))
    assert [type(m) for m in back] == [HumanMessage, AIMessage, ToolMessage]
    assert back[1].tool_calls[0]["args"] == {"a": 2, "b": 3}


async def test_streamed_tool_calls_merge_into_tool_calls() -> None:
    call = ModelToolCall("c9", "multiply", '{"a": 1, "b": 2}')
    role = _Recording(_answer("", call, reasoning="hmm"), _answer("bad", ModelToolCall("c10", "multiply", "not json")))
    llm = chat_model(role).bind_tools([multiply])
    chunks = [c async for c in llm.astream("x")]
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    assert merged.tool_calls == [{"name": "multiply", "args": {"a": 1, "b": 2}, "id": "c9", "type": "tool_call"}]
    assert merged.content == "" and merged.additional_kwargs["reasoning_content"] == "hmm"
    bad = await llm.ainvoke("y")
    assert bad.tool_calls == [] and bad.invalid_tool_calls[0]["id"] == "c10"


async def test_stop_sequences_are_refused() -> None:
    llm = chat_model(_Recording(_answer("x")))
    with pytest.raises(ValueError, match="stop sequences"):
        await llm.ainvoke("hi", stop=["\n"])


async def test_react_agent_runs_on_a_role() -> None:
    models = FakeModels({"r": [ToolCall("multiply", {"a": 3, "b": 5}), "Fifteen."]})
    agent = create_react_agent(chat_model(models.role("r")), tools=[multiply], prompt="You do math.")
    state = await agent.ainvoke({"messages": [HumanMessage("3 times 5?")]})
    assert [type(m).__name__ for m in state["messages"]] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert state["messages"][-1].content == "Fifteen."
    assert models.calls("r")[0].system == "You do math."


# ---------------------------------------------------------------------- tools


class _P:
    id = name = "ana"


async def test_langchain_tools_inject_p_and_turn() -> None:
    seen: dict[str, Any] = {}

    @tool
    async def whoami(p, turn, greeting: str, punct: str = "!") -> str:
        """Greet the participant."""
        seen["turn"] = turn
        return f"{greeting}, {p.name}{punct}"

    @tool
    async def facts(p) -> dict:
        """Facts about the participant."""
        return {"name": p.name}

    greet, info = langchain_tools(_P(), whoami, facts)
    assert set(greet.tool_call_schema.model_json_schema()["properties"]) == {"greeting", "punct"}
    token = CURRENT_TURN.set("the turn")  # type: ignore[arg-type]
    try:
        assert await greet.ainvoke({"greeting": "hi"}) == "hi, ana!"
    finally:
        CURRENT_TURN.reset(token)
    assert seen["turn"] == "the turn"
    assert await info.ainvoke({}) == '{"name":"ana"}'
    with pytest.raises(ValueError, match="duplicate"):
        langchain_tools(_P(), whoami, whoami)


async def test_from_langchain_is_a_conversation_tool() -> None:
    converted = from_langchain(multiply)
    assert converted.name == "multiply" and converted.description == "Multiply two integers."
    assert set(converted.request_model.model_json_schema()["properties"]) == {"a", "b"}
    assert await converted.handler(converted.request_model(a=4, b=5)) == "20"
    Conversation(role=FakeModels().role("r"), output=TextSink(), instructions="x", tools=[converted])
    with pytest.raises(TypeError):
        from_langchain(lambda: None)  # type: ignore[arg-type]


async def test_return_direct_carries_over_both_ways() -> None:
    @tool(return_direct=True)
    async def reading() -> str:
        """The meter reading, said as it is."""
        return "7 C"

    @lc_tool(return_direct=True)
    def status() -> str:
        """Whether the monitor runs."""
        return "running"

    (direct,) = langchain_tools(_P(), reading)
    assert direct.return_direct and await direct.ainvoke({}) == "7 C"
    assert not langchain_tools(_P(), _plain)[0].return_direct
    assert from_langchain(status).return_direct and not from_langchain(multiply).return_direct


@tool
async def _plain() -> str:
    """A plain tool."""
    return "x"


# ---------------------------------------------------------------------- messages and stream_text


async def test_messages_is_one_system_message_of_the_non_empty_channels() -> None:
    notes = ContextLog("notes", scope="app", budget_bytes=500)
    mode = ContextValue("mode", scope="app", budget_bytes=100)
    empty = ContextValue("empty", scope="app", budget_bytes=100)
    assert await messages(None, notes, empty) == []
    await notes.publish(None, "buy milk")
    await mode.set(None, "dictation")
    (system,) = await messages(None, notes, empty, mode)
    assert isinstance(system, SystemMessage)
    assert system.content == "# Context\n\n## notes\nbuy milk\n\n## mode\ndictation"


async def test_stream_text_keeps_only_ai_text() -> None:
    async def source():
        yield AIMessageChunk(content="Hel"), {"langgraph_node": "agent"}
        yield AIMessageChunk(content="", tool_call_chunks=[{"name": "x", "args": "{}", "id": "1", "index": 0}]), {}
        yield ToolMessage("42", tool_call_id="1"), {}
        yield "updates", {"agent": {}}
        yield "messages", (AIMessageChunk(content="lo"), {})
        yield AIMessageChunk(content=[{"type": "reasoning", "reasoning": "hmm"}, {"type": "text", "text": ", "}])
        yield AIMessage(content="ignored", tool_calls=[{"name": "x", "args": {}, "id": "2"}])
        yield HumanMessage("not me")
        yield ("sub:1",), ("messages", (AIMessageChunk(content="world"), {}))
        yield AIMessageChunk(content="")
        yield "!"

    assert [t async for t in stream_text(source())] == ["Hel", "lo", ", ", "world", "!"]


async def test_stream_text_of_a_react_agent_speaks_the_answer_once() -> None:
    models = FakeModels({"r": [ToolCall("multiply", {"a": 2, "b": 4}), "It is eight."]})
    agent = create_react_agent(chat_model(models.role("r")), tools=[multiply])
    stream = agent.astream({"messages": [HumanMessage("2 times 4?")]}, stream_mode="messages")
    assert "".join([t async for t in stream_text(stream)]) == "It is eight."


async def test_stream_text_of_a_react_agent_streams_each_token_once() -> None:
    call = ModelToolCall("c1", "multiply", '{"a": 2, "b": 4}')
    role = _Recording(_answer("Let me check.", call), _answer("It is eight."))
    agent = create_react_agent(chat_model(role), tools=[multiply])
    stream = agent.astream({"messages": [HumanMessage("2 times 4?")]}, stream_mode="messages")
    # text before a tool call streams too (chunks without tool calls); the final message is not re-emitted
    assert [t async for t in stream_text(stream)] == ["Let ", "me ", "check. ", "It ", "is ", "eight. "]
    assert role.calls[1]["messages"][-1].content == "8"


# ---------------------------------------------------------------------- GraphBackend


class _State(TypedDict):
    request: str
    notes: Annotated[list[str], add]
    answer: str


def _graph(gate: asyncio.Event | None = None, cancelled: list[str] | None = None):
    async def gather(state: _State) -> dict:
        get_stream_writer()({"progress": f"reading about {state['request']}"})
        get_stream_writer()({"other": "not progress"})
        if gate is not None:
            try:
                await gate.wait()
            except asyncio.CancelledError:
                assert cancelled is not None
                cancelled.append("gather")
                raise
        return {"notes": ["a"]}

    async def more(state: _State) -> dict:
        return {"notes": ["b"]}

    async def write(state: _State) -> dict:
        return {"answer": " + ".join(state["notes"])}

    graph = StateGraph(_State)
    for name, fn in (("gather", gather), ("more", more), ("write", write)):
        graph.add_node(name, fn)
    graph.add_edge(START, "gather")
    graph.add_edge("gather", "more")
    graph.add_edge("more", "write")
    graph.add_edge("write", END)
    return graph.compile()


def _job() -> Job:
    return Job(id="j1", request="tea", participant="ana")


async def test_graph_backend_progress_and_result() -> None:
    backend = GraphBackend(_graph(), input=lambda r: {"request": r, "notes": []}, result=lambda s: s["answer"])
    updates = [u async for u in backend.run("tea", _job())]
    assert [(u.kind, u.text) for u in updates] == [
        ("progress", "reading about tea"),  # only the graph's own words: finished nodes say nothing by default
        ("result", "a + b"),  # the reducer's view of the state, not the last update
    ]
    assert not any(name in u.text for u in updates for name in ("gather", "more", "write"))


async def test_graph_backend_custom_progress_and_speak() -> None:
    backend = GraphBackend(
        _graph(),
        input=lambda r: {"request": r, "notes": []},
        progress=lambda node, update: f"{node}: {update}" if node == "more" else None,
        result=lambda s: JobUpdate("result", s["answer"], speak="Done."),
    )
    updates = [u async for u in backend.run("tea", _job())]
    assert [u.text for u in updates] == ["reading about tea", "more: {'notes': ['b']}", "a + b"]
    assert updates[-1].speak == "Done."


async def test_graph_backend_cancel_cancels_the_running_node() -> None:
    cancelled: list[str] = []
    backend = GraphBackend(_graph(asyncio.Event(), cancelled), input=lambda r: {"request": r, "notes": []})
    seen: list[JobUpdate] = []

    async def consume() -> None:
        async for update in backend.run("tea", _job()):
            seen.append(update)

    task = asyncio.create_task(consume())
    for _ in range(200):
        if seen:
            break
        await asyncio.sleep(0.01)
    assert [u.text for u in seen] == ["reading about tea"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled == ["gather"]


# ---------------------------------------------------------------------- samples 08 and 09


def _reasoning_for_08(call) -> Any:
    """One reply per call to ``answer_with_reasoning``, by prompt: the graph's three kinds, else the conversation."""
    if call.prompt.startswith("List up to three"):
        return "Which cafes are near?\nWhich are quiet?"
    if call.prompt.startswith("Answer briefly"):
        return "Cafe A." if "near" in call.prompt else "Cafe A is quiet."
    if call.prompt.startswith("Request:"):
        assert "Which are quiet?: Cafe A is quiet." in call.prompt
        return "Cafe A, by the station, is quiet."
    if "Tool start_research returned" in call.prompt:
        return "I'm on it."
    return ToolCall("start_research", {"request": "quiet cafe near the station"})


async def test_sample_08_delegate_langgraph(load_sample) -> None:
    m = load_sample("08-delegate-langgraph")
    m.research._progress_every_s = None
    models = FakeModels(
        {"answer_with_reasoning": [_reasoning_for_08] * 6, "answer_lightweight": ["Cafe A is quiet and close."]}
    )
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("find me a quiet cafe near the station")
        await until(s, lambda: len(s.spoken()) == 2)
        assert s.spoken() == ["I'm on it.", "Cafe A is quiet and close."]
        p = s.participant()
        (job,) = m.research.jobs(p)
        assert job.state == "done" and job.result_text == "Cafe A, by the station, is quiet."
        assert "Cafe A, by the station" in await m.research.results.view(p)
        progress = [e.data["text"] for e in s.trace.events(kind="job") if e.data["phase"] == "progress"]
        assert progress == ["Looking into: Which cafes are near?", "Looking into: Which are quiet?"]  # no node names
        graph_calls = [e for e in s.trace.events(kind="role") if e.data.get("via") == "langchain"]
        assert len(graph_calls) == 4 and {e.name for e in graph_calls} == {"answer_with_reasoning"}


async def test_sample_08_cancel_stops_the_graph(load_sample) -> None:
    m = load_sample("08-delegate-langgraph")
    cancelled: list[str] = []
    m.research._backend = GraphBackend(_graph(asyncio.Event(), cancelled), input=lambda r: {"request": r, "notes": []})
    models = FakeModels(
        {
            "answer_with_reasoning": [
                ToolCall("start_research", {"request": "x"}),
                "Started.",
                ToolCall("cancel_research", {}),
                "Cancelled it.",
            ]
        }
    )
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("research x")
        await until(s, lambda: "reading about x" in m.research.status.get(s.participant()))
        await s.say("never mind")
        (job,) = m.research.jobs(s.participant())
        # the graph's own task unwinds over a few loop turns, which ``settle`` does not wait for
        await until(s, lambda: len(s.spoken()) == 2)
        assert job.state == "cancelled" and cancelled == ["gather"]
        assert s.spoken() == ["Started.", "Cancelled it."]


async def test_sample_09_math_agent_with_tools(load_sample) -> None:
    m = load_sample("09-langchain-agent")
    models = FakeModels(
        {
            "decide": ["math", "math"],
            "answer_with_reasoning": [
                ToolCall("convert", {"value": 10, "unit_from": "km", "unit_to": "mi"}),
                "Ten kilometers is about 6.21 miles.",
                ToolCall("look", {"question": "how long is the ruler?"}),
                "The ruler is 30 centimeters.",
            ],
            "perceive": ["A 30 cm ruler."],
        }
    )
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("how many miles is 10 km?")
        # LangChain runs a sync tool in a thread, which ``settle`` takes for idle: wait for the answer instead
        await until(s, lambda: len(s.spoken()) == 1)
        assert s.spoken() == ["Ten kilometers is about 6.21 miles."]
        first = models.calls("answer_with_reasoning")[1]
        assert "Tool convert returned: 6.21 mi" in first.prompt

        await s.show_frame(solid_frame("white"))
        await s.say("how long is this ruler?")
        await until(s, lambda: len(s.spoken()) == 2)
        assert s.spoken()[-1] == "The ruler is 30 centimeters."
        (looked,) = models.calls("perceive")
        assert looked.prompt == "how long is the ruler?"
        assert [e.name for e in s.trace.events(kind="tool")] == ["look"]
        # the second turn saw the first in its history, after the agent's prompt
        second = models.calls("answer_with_reasoning")[2]
        assert second.system.startswith("You solve math and unit questions.")
        assert "Ten kilometers is about 6.21 miles." in second.system


async def test_sample_09_conversation_uses_the_langchain_tool(load_sample) -> None:
    m = load_sample("09-langchain-agent")
    models = FakeModels(
        {
            "decide": ["conversation"],
            "answer_with_reasoning": [
                ToolCall("convert", {"value": 100, "unit_from": "c", "unit_to": "f"}),
                "That's 212 degrees Fahrenheit.",
            ],
        }
    )
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("what is 100 C in F?")
        await until(s, lambda: len(s.spoken()) == 1)
        assert s.spoken() == ["That's 212 degrees Fahrenheit."]
        (used,) = s.trace.events(kind="tool")
        assert used.name == "convert" and used.data["result"] == "212.00 f"
