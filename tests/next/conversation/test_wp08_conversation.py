# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversation parts that outlived the first-round conversation: the ``@tool`` checks.

No ``from __future__ import annotations``: ``@tool`` reads parameter annotations.
"""

from typing import Annotated

import pytest
from pydantic import BaseModel, Field
from xr_ai_kit import Conversation
from xr_ai_next import TextSink
from xr_ai_next._base.participant import Participant, Turn
from xr_ai_next._base.testing import FakeModels
from xr_ai_next._base.tools import _spec, tool

# ---- tools ----------------------------------------------------------------------------------------------


class Saved(BaseModel):
    count: int


async def test_tool_receives_p_and_turn_hides_them_from_the_schema_and_feeds_the_next_round() -> None:
    @tool
    async def save_note(
        p: Participant, turn: Turn, text: Annotated[str, Field(description="what to remember")], times: int = 1
    ) -> Saved:
        "Save a note the user wants to remember."
        return Saved(count=times)

    spec = _spec(save_note)
    assert spec.wants_p and spec.wants_turn
    schema = spec.request_model.model_json_schema()
    assert set(schema["properties"]) == {"text", "times"} and schema["properties"]["text"]["description"]


async def test_a_plain_function_and_a_bad_tool_are_checked_when_the_conversation_is_built() -> None:
    async def undocumented(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="needs a description"):
        tool(undocumented)
    with pytest.raises(ValueError, match="duplicate"):
        Conversation(
            role=FakeModels().role("answer"), output=TextSink(), instructions="x", tools=[_named("a"), _named("a")]
        )


def _named(name):
    async def f() -> str:
        "d"
        return "d"

    return tool(name=name)(f)


# ---- return_direct: a tool's result is the answer; the loop says it, the tool never speaks ---------------


async def _answer(conversation: Conversation, text: str = "what does the meter say?") -> Participant:
    p = Participant("p")
    await conversation.answer(Turn(text=text, timestamp_us=0, id="t1"), p)
    return p


async def test_a_direct_tools_result_is_the_answer_and_ends_the_tool_loop() -> None:
    from xr_ai_next.testing import ToolCall

    sink = TextSink()

    @tool(return_direct=True)
    async def read_meter() -> str:
        "Read the meter."
        return "The meter reads 7."

    @tool
    async def lookup() -> str:
        "Look something up for the model."
        return "seven"

    models = FakeModels({"answer": [ToolCall("lookup"), ToolCall("read_meter"), "never asked"]})
    conversation = Conversation(role=models.role("answer"), output=sink, instructions="x", tools=[lookup, read_meter])
    await _answer(conversation)
    # lookup's result went back to the model; read_meter's result is the answer: no third round, no fallback
    assert sink.said["p"] == ["The meter reads 7."]
    assert len(models.calls("answer")) == 2


async def test_a_direct_tool_may_return_a_stream_which_is_said_as_it_arrives() -> None:
    from xr_ai_next.testing import ToolCall

    sink = TextSink()
    seen: list[str] = []

    async def chunks():
        for part in ("The view ", "shows ", "a sign."):
            seen.append(part)
            yield part

    @tool(return_direct=True)
    async def current_view():
        "Describe the view."
        return chunks()

    models = FakeModels({"answer": [ToolCall("current_view")]})
    conversation = Conversation(role=models.role("answer"), output=sink, instructions="x", tools=[current_view])
    await _answer(conversation)
    assert sink.said["p"] == ["The view shows a sign."] and seen == ["The view ", "shows ", "a sign."]


async def test_a_direct_tool_that_fails_before_answering_goes_back_to_the_model() -> None:
    from xr_ai_next.testing import ToolCall

    sink = TextSink()

    @tool(return_direct=True)
    async def read_meter() -> str:
        "Read the meter."
        raise RuntimeError("no camera")

    models = FakeModels({"answer": [ToolCall("read_meter"), "I can't see the meter right now."]})
    conversation = Conversation(role=models.role("answer"), output=sink, instructions="x", tools=[read_meter])
    await _answer(conversation)
    assert sink.said["p"] == ["I can't see the meter right now."]
    assert "no camera" in models.calls("answer")[1].prompt  # the model read the failure


async def test_an_empty_direct_answer_ends_the_turn_with_nothing_more_said() -> None:
    from xr_ai_next.testing import ToolCall

    sink = TextSink()

    @tool(return_direct=True)
    async def announce(p: Participant) -> None:
        "Tell everyone in the room."
        await p.say("Ana checked the valve.", to=sink)  # the tool spoke for itself

    @tool(return_direct=True)
    async def quiet() -> str:
        "Nothing to add."
        return ""

    models = FakeModels({"answer": [ToolCall("announce"), ToolCall("quiet"), "never asked"]})
    for name in ("announce", "quiet"):
        conversation = Conversation(
            role=models.role("answer"), output=sink, instructions="x", tools=[announce, quiet], fallback="Sorry."
        )
        await _answer(conversation, f"use {name}")
    said = [text for text in sink.said["p"] if text]
    assert said == ["Ana checked the valve."]  # no fallback, no second answer
    assert len(models.calls("answer")) == 2


async def test_a_direct_stream_that_breaks_before_saying_anything_falls_back() -> None:
    from xr_ai_next.testing import ToolCall

    sink = TextSink()

    async def broken():
        raise RuntimeError("stream lost")
        yield ""  # pragma: no cover - makes this an async generator

    async def first_then_broken():
        yield ""
        async for part in broken():
            yield part

    @tool(return_direct=True)
    async def current_view():
        "Describe the view."
        return first_then_broken()

    models = FakeModels({"answer": [ToolCall("current_view")]})
    conversation = Conversation(
        role=models.role("answer"), output=sink, instructions="x", tools=[current_view], fallback="Sorry."
    )
    await _answer(conversation)
    assert sink.said["p"] == ["Sorry."]


async def test_tool_specs_carry_return_direct_and_join_a_streamed_result_for_the_model() -> None:
    from xr_ai_next.tools import Tool, describe

    async def parts():
        yield "a"
        yield "b"

    @tool
    async def streamed():
        "A streamed result."
        return parts()

    class NoArguments(BaseModel):
        pass

    foundation = Tool("f", "A foundation tool.", NoArguments, None, lambda _r: "x", return_direct=True)
    assert not describe(streamed).return_direct and describe(foundation).return_direct
    assert await describe(streamed).call(Participant("p"), {}) == "ab"


async def test_a_direct_foundation_tool_is_said_with_its_own_renderer() -> None:
    from xr_ai_next.testing import ToolCall
    from xr_ai_next.tools import Tool

    class NoArguments(BaseModel):
        pass

    class Reading(BaseModel):
        text: str

    sink = TextSink()
    meter = Tool(
        "meter",
        "Read the meter.",
        NoArguments,
        Reading,
        lambda _r: Reading(text="The meter reads 7."),
        return_direct=True,
        render_result=lambda r: r.text,
    )
    models = FakeModels({"answer": [ToolCall("meter")]})
    await _answer(Conversation(role=models.role("answer"), output=sink, instructions="x", tools=[meter]))
    assert sink.said["p"] == ["The meter reads 7."]  # not its JSON


async def test_a_foundation_tool_with_the_default_renderer_is_read_as_before() -> None:
    from xr_ai_next.tools import Tool, describe

    class NoArguments(BaseModel):
        pass

    plain = Tool("facts", "Some facts.", NoArguments, None, lambda _r: {"rooms": 2})
    assert await describe(plain).call(Participant("p"), {}) == '{"rooms":2}'  # JSON of a dict, as render() gives


async def test_a_direct_stream_cut_off_by_cancelling_the_turn_closes_its_source() -> None:
    import asyncio

    from xr_ai_next.testing import ToolCall

    started, closed = asyncio.Event(), asyncio.Event()

    async def view():
        try:
            yield "The view "
            started.set()
            await asyncio.Event().wait()  # the rest never comes
            yield "never"
        finally:
            closed.set()

    @tool(return_direct=True)
    async def current_view():
        "Describe the view."
        return view()

    models = FakeModels({"answer": [ToolCall("current_view")]})
    conversation = Conversation(role=models.role("answer"), output=TextSink(), instructions="x", tools=[current_view])
    task = asyncio.ensure_future(_answer(conversation))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()  # closed by the cancellation itself, not later by the garbage collector
