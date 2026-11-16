# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FakeModels: scripted replies, recording, latency and cancellation."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel
from xr_ai_app._base.testing import FakeModels, ScriptExhausted


class Verdict(BaseModel):
    ok: bool


async def test_ask_replays_script_in_order_and_records_calls() -> None:
    models = FakeModels({"answer": ["first", "second"]})
    role = models.role("answer")
    assert role.name == "answer"
    assert await role.ask(None, "q1") == "first"
    assert await role.ask(None, "q2") == "second"
    assert [c.prompt for c in models.calls("answer")] == ["q1", "q2"]
    assert all(c.finished and not c.cancelled for c in models.calls("answer"))
    assert models.pending_replies("answer") == 0


async def test_exhausted_script_fails_loudly() -> None:
    models = FakeModels({"answer": ["only"]})
    role = models.role("answer")
    await role.ask(None, "q")
    with pytest.raises(ScriptExhausted):
        await role.ask(None, "again")
    with pytest.raises(ScriptExhausted):
        await models.role("never-scripted").ask(None, "q")


async def test_structured_reply_is_validated_and_exceptions_are_raised() -> None:
    models = FakeModels({"decide": [{"ok": True}, Verdict(ok=False), ValueError("backend down")]})
    role = models.role("decide")
    assert await role.ask(None, "q", response_format=Verdict) == Verdict(ok=True)
    assert await role.ask(None, "q", response_format=Verdict) == Verdict(ok=False)
    with pytest.raises(ValueError, match="backend down"):
        await role.ask(None, "q")
    assert models.calls("decide")[0].response_format is Verdict


async def test_decide_checks_the_choice_and_records_context() -> None:
    models = FakeModels({"decide": ["yes", "maybe"]})
    role = models.role("decide")
    assert await role.decide("ready?", choices=["yes", "no"], context="kettle") == "yes"
    call = models.calls("decide")[0]
    assert (call.method, call.choices, call.context) == ("decide", ("yes", "no"), "kettle")
    with pytest.raises(ValueError, match="not one of"):
        await role.decide("ready?", choices=["yes", "no"])


async def test_callable_reply_sees_the_call() -> None:
    models = FakeModels({"answer": [lambda call: f"echo {call.prompt}"]})
    assert await models.role("answer").ask(None, "hi") == "echo hi"


async def test_stream_yields_chunks_that_join_to_the_reply() -> None:
    models = FakeModels({"answer": ["one two three"]})
    chunks = [c async for c in models.role("answer").stream(None, "q")]
    assert len(chunks) == 3
    assert "".join(chunks) == "one two three"
    call = models.calls("answer")[0]
    assert call.finished and call.close_count == 0 and not call.cancelled


async def test_latency_delays_the_reply() -> None:
    models = FakeModels({"answer": ["slow"]})
    models.set_latency("answer", 0.1)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await models.role("answer").ask(None, "q")
    assert loop.time() - start >= 0.09


async def test_cancelling_an_ask_is_recorded_and_frees_the_call() -> None:
    models = FakeModels({"answer": ["never"]})
    models.set_latency("answer", 3600)
    task = asyncio.create_task(models.role("answer").ask(None, "q"))
    await models.wait_started("answer")
    assert models.active("answer") == 1 and models.any_active()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert models.cancelled("answer") == 1
    assert models.active("answer") == 0
    assert models.pending_replies("answer") == 1  # a cancelled call consumed no reply


async def test_cancelling_a_stream_closes_it_exactly_once() -> None:
    models = FakeModels({"answer": ["one two three four"]})
    models.set_latency("answer", 0.0, chunk_s=3600)

    async def consume() -> None:
        async for _ in models.role("answer").stream(None, "q"):
            pass

    task = asyncio.create_task(consume())
    await models.wait_started("answer")
    await asyncio.sleep(0.01)  # first chunk delivered, second is sleeping
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    call = models.calls("answer")[0]
    assert call.cancelled and not call.finished
    assert call.close_count == 1
    assert models.active("answer") == 0


async def test_explicit_aclose_counts_once_and_a_finished_stream_is_not_a_cancel() -> None:
    models = FakeModels({"answer": ["a b c", "d e"]})
    role = models.role("answer")
    stream = role.stream(None, "q1")
    assert await anext(stream) == "a "
    await stream.aclose()
    await stream.aclose()
    first = models.calls("answer")[0]
    assert first.cancelled and first.close_count == 1

    done = role.stream(None, "q2")
    assert [c async for c in done] == ["d ", "e"]
    await done.aclose()
    second = models.calls("answer")[1]
    assert second.finished and not second.cancelled and second.close_count == 0


async def test_wait_started_times_out_when_no_call_arrives() -> None:
    models = FakeModels()
    with pytest.raises(TimeoutError):
        await models.wait_started("answer", timeout_s=0.05)


async def test_ask_and_stream_options_and_several_frames_are_recorded() -> None:
    from xr_ai_app._base.frames import Frame

    models = FakeModels({"perceive": ["a", "b b"]})
    role = models.role("perceive")
    f1 = Frame(image=b"1", timestamp_us=1, width=1, height=1)
    f2 = Frame(image=b"2", timestamp_us=2, width=1, height=1)
    await role.ask([f1, f2], "diff", system="sys", thinking=False, max_tokens=5, temperature=0.2, timeout_s=3)
    [x async for x in role.stream(f1, "again", system="s2")]
    first, second = models.calls("perceive")
    assert first.frames == (f1, f2) and first.frame is f1 and first.system == "sys"
    assert (first.thinking, first.max_tokens, first.temperature, first.timeout_s) == (False, 5, 0.2, 3)
    assert second.frames == (f1,) and second.system == "s2" and second.timeout_s is None


async def test_fake_timeout_and_structured_output_failure_match_the_contract() -> None:
    from xr_ai_app._base.profiles import StructuredOutputError

    models = FakeModels({"answer": ["slow", {"nope": 1}]})
    models.set_latency("answer", 1.0)
    role = models.role("answer")
    with pytest.raises(TimeoutError):
        await role.ask(None, "q", timeout_s=0.01)
    assert models.active("answer") == 0
    models.set_latency("answer", 0.0)
    with pytest.raises(StructuredOutputError):
        await role.ask(None, "q", response_format=Verdict)


async def test_decide_takes_criteria_and_frames_and_returns_a_decision() -> None:
    from xr_ai_app._base.frames import Frame
    from xr_ai_app._base.profiles import Decision

    models = FakeModels({"decide": ["yes", Decision("no", confidence=0.8, probabilities={"yes": 0.2, "no": 0.8})]})
    role = models.role("decide")
    frame = Frame(image=b"1", timestamp_us=1, width=1, height=1)
    first = await role.decide("relevant?", choices={"yes": "on topic", "no": "off topic"}, frames=[frame])
    assert first == "yes" and isinstance(first, Decision) and first.confidence is None and first.probabilities is None
    call = models.calls("decide")[0]
    assert call.criteria == {"yes": "on topic", "no": "off topic"} and call.frames == (frame,)
    second = await role.decide("relevant?", choices=["yes", "no"])
    assert second == "no" and second.confidence == 0.8 and second.probabilities["yes"] == 0.2
