# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant: state, output, spawn and the artifact sink (rules R1, R6.1)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel
from xr_ai_interaction import ResponseOutput
from xr_ai_next._base.artifacts import ArtifactSink
from xr_ai_next._base.participant import CURRENT_TURN, Participant, ParticipantState, Turn
from xr_ai_next._base.scope import ParticipantScope
from xr_ai_next._base.topics import OUTPUT
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe


class _Collector(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[tuple[str | None, ResponseOutput]] = []

    @subscribe(OUTPUT)
    async def collect(self, message: ResponseOutput, ctx: RuntimeContext) -> None:
        self.outputs.append((ctx.metadata.participant_id, message))


async def _bound(tmp_path: Path | None = None) -> tuple[Participant, _Collector, AgentRuntime]:
    runtime = AgentRuntime()
    collector = runtime.register("collector", _Collector())
    await runtime.start()
    p = Participant("p1")
    p._bind(
        scope=ParticipantScope("p1"),
        runtime=runtime,
        artifacts=ArtifactSink(tmp_path, participant_id="p1") if tmp_path is not None else None,
    )
    return p, collector, runtime


def test_state_is_a_mapping_and_attributes_and_missing_attributes_raise() -> None:
    state = ParticipantState()
    state.step = 2
    state["name"] = "tea"
    assert state["step"] == 2 and state.name == "tea"
    assert dict(state) == {"step": 2, "name": "tea"} and len(state) == 2
    del state.step
    assert "step" not in state
    with pytest.raises(AttributeError):
        state.missing  # noqa: B018
    with pytest.raises(KeyError):
        state["missing"]


async def test_say_publishes_an_update_for_this_participant() -> None:
    p, collector, runtime = await _bound()
    await p.say("hello", urgent=True)
    ((pid, message),) = collector.outputs
    assert pid == "p1"
    assert (message.kind, message.text, message.urgent) == ("update", "hello", True)
    await runtime.stop()


async def test_reply_in_a_turn_is_a_reply_to_that_turn_and_streams_fragments() -> None:
    p, collector, runtime = await _bound()
    CURRENT_TURN.set(Turn(text="q", timestamp_us=5, id="t1"))

    async def fragments() -> AsyncIterator[str]:
        yield "one "
        yield "two"

    await p.reply(fragments())
    kinds = [(m.kind, m.text, m.request_id, m.final) for _, m in collector.outputs]
    assert kinds == [
        ("reply", "one ", "t1", False),
        ("reply", "two", "t1", False),
        ("reply", "", "t1", True),
    ]
    assert len({m.response_id for _, m in collector.outputs}) == 1
    await runtime.stop()
    CURRENT_TURN.set(None)


async def test_reply_outside_a_turn_is_spoken_as_an_update() -> None:
    p, collector, runtime = await _bound()
    await p.reply("hi")
    assert [(m.kind, m.text) for _, m in collector.outputs] == [("update", "hi")]
    await runtime.stop()


async def test_output_is_silent_after_the_scope_stops_admitting() -> None:
    p, collector, runtime = await _bound()
    await p._scope.leave()
    await p.say("late")
    await p.reply("late")
    assert collector.outputs == []
    await runtime.stop()


async def test_output_after_the_runtime_closed_never_raises_into_app_code() -> None:
    p, _, runtime = await _bound()
    await runtime.stop()
    await p.say("late")
    await p.reply("late")


async def test_spawn_and_cancel_delegate_to_the_scope() -> None:
    p, _, runtime = await _bound()

    async def blocked() -> None:
        await asyncio.sleep(10)

    task = p.spawn(blocked(), key="k")
    await asyncio.sleep(0)
    assert await p.cancel("k") is True and task.cancelled()
    await runtime.stop()


async def test_release_drains_artifacts_then_clears_state() -> None:
    p, _, runtime = await _bound(Path("."))
    p.state.keep = 1
    p.session.state.keep = 2
    await p._release()
    assert dict(p.state) == {} and dict(p.session.state) == {}
    assert p._artifacts is not None and p._artifacts._closed
    await runtime.stop()


async def test_artifact_sink_writes_jsonl_and_blobs_in_order(tmp_path: Path) -> None:
    sink = ArtifactSink(tmp_path, participant_id="p/1")

    class Row(BaseModel):
        n: int

    await asyncio.gather(*(sink.write_jsonl("rows", Row(n=i)) for i in range(5)))
    await sink.write_jsonl("rows", {"n": 5})
    path = await sink.write_blob("frame.jpg", b"\xff\xd8", content_type="image/jpeg")
    lines = (sink.directory / "rows.jsonl").read_text().splitlines()
    assert [json.loads(line)["n"] for line in lines] == [0, 1, 2, 3, 4, 5]
    assert path == sink.directory / "frame.jpg" and path.read_bytes() == b"\xff\xd8"
    assert sink.directory.parent == tmp_path and "/" not in sink.directory.name


async def test_artifact_sink_drain_waits_for_accepted_writes_then_later_writes_are_noops(tmp_path: Path) -> None:
    sink = ArtifactSink(tmp_path, participant_id="p1")
    writers = [asyncio.ensure_future(sink.write_jsonl("rows", {"n": i})) for i in range(20)]
    await asyncio.sleep(0)
    await sink.drain()
    await asyncio.gather(*writers)
    assert len((sink.directory / "rows.jsonl").read_text().splitlines()) == 20
    await sink.write_jsonl("rows", {"n": 99})
    assert (await sink.write_blob("late.bin", b"x")) == sink.directory / "late.bin"
    assert not (sink.directory / "late.bin").exists()
    assert len((sink.directory / "rows.jsonl").read_text().splitlines()) == 20


async def test_a_participant_without_artifacts_dir_explains_itself() -> None:
    with pytest.raises(RuntimeError, match="artifacts_dir"):
        Participant("p1").artifacts  # noqa: B018


class _ClosingSubscriber(Agent):
    """An OUTPUT subscriber that fails the way a closing consumer does."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    @subscribe(OUTPUT)
    async def collect(self, message: ResponseOutput, ctx: RuntimeContext) -> None:
        raise self.error


async def test_a_closing_subscriber_never_raises_into_say_or_reply() -> None:
    from xr_ai_runtime import RuntimeClosedError

    runtime = AgentRuntime()
    runtime.register("closing", _ClosingSubscriber(RuntimeClosedError("voice closed")))
    await runtime.start()
    p = Participant("p1")
    p._bind(scope=ParticipantScope("p1"), runtime=runtime, artifacts=None)
    await p.say("hello")
    await p.reply("hello")
    await runtime.stop()


async def test_a_failing_subscriber_that_is_not_a_closing_error_still_surfaces() -> None:
    runtime = AgentRuntime()
    runtime.register("broken", _ClosingSubscriber(ValueError("bug")))
    await runtime.start()
    p = Participant("p1")
    p._bind(scope=ParticipantScope("p1"), runtime=runtime, artifacts=None)
    with pytest.raises(BaseExceptionGroup):
        await p.say("hello")
    await runtime.stop()
