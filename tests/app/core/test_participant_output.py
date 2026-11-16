# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant amendments: the say source, withdraw, and later additions."""

from __future__ import annotations

from xr_ai_app._base.participant import Participant
from xr_ai_app._base.scope import ParticipantScope
from xr_ai_app._base.topics import OUTPUT
from xr_ai_interaction import ResponseOutput
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe


class _Collector(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[str, str, str]] = []  # (producer, kind, text)

    @subscribe(OUTPUT)
    async def collect(self, message: ResponseOutput, ctx: RuntimeContext) -> None:
        self.seen.append((ctx.metadata.source, message.kind, message.text))


async def _bound() -> tuple[Participant, _Collector, AgentRuntime]:
    runtime = AgentRuntime()
    collector = runtime.register("collector", _Collector())
    await runtime.start()
    p = Participant("p1")
    p._bind(scope=ParticipantScope("p1"), runtime=runtime, artifacts=None)
    return p, collector, runtime


async def test_sourced_say_replaces_its_queued_utterance_and_leaves_others_alone() -> None:
    p, collector, runtime = await _bound()
    await p.say("plain")
    await p.say("first", source="boil")
    await p.say("second", source="boil")
    assert collector.seen == [
        ("app", "update", "plain"),
        ("app:boil", "withdraw", ""),
        ("app:boil", "update", "first"),
        ("app:boil", "withdraw", ""),  # the consumer drops "first" (same producer) before "second" queues
        ("app:boil", "update", "second"),
    ]
    await runtime.stop()


async def test_withdraw_removes_only_that_source_and_reports_whether_one_was_queued() -> None:
    p, collector, runtime = await _bound()
    assert await p.withdraw("boil") is False  # nothing queued: nothing published
    await p.say("watch out", source="boil")
    assert await p.withdraw("boil") is True
    assert await p.withdraw("boil") is False
    assert collector.seen[-1] == ("app:boil", "withdraw", "")
    assert [s for s, _, _ in collector.seen] == ["app:boil", "app:boil", "app:boil"]
    await runtime.stop()


async def test_say_source_and_withdraw_are_silent_after_the_scope_stopped_admitting() -> None:
    p, collector, runtime = await _bound()
    await p.say("queued", source="a")
    await p._scope.leave()
    before = list(collector.seen)
    await p.say("late", source="a")
    assert await p.withdraw("a") is False
    assert collector.seen == before
    await runtime.stop()
    await p.say("later", source="a")  # runtime closed: still no raise


async def test_the_voice_output_bridge_keeps_the_producer_so_withdraw_stays_per_source() -> None:
    from xr_ai_app._base.app import _OutputBridge
    from xr_ai_voice import VOICE_OUTPUT_TOPIC

    class Voice(Agent):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[tuple[str | None, str, str]] = []

        @subscribe(VOICE_OUTPUT_TOPIC)
        async def collect(self, message: ResponseOutput, ctx: RuntimeContext) -> None:
            self.seen.append((ctx.metadata.participant_id, ctx.metadata.source, message.kind))

    runtime = AgentRuntime()
    voice = runtime.register("voice", Voice())
    runtime.register("bridge", _OutputBridge(runtime))
    await runtime.start()
    p = Participant("p1")
    p._bind(scope=ParticipantScope("p1"), runtime=runtime, artifacts=None)
    await p.say("a", source="boil")
    await p.withdraw("boil")
    await runtime.stop()
    assert voice.seen == [
        ("p1", "app:boil", "withdraw"),
        ("p1", "app:boil", "update"),
        ("p1", "app:boil", "withdraw"),
    ]


async def test_reply_returns_the_spoken_text_inside_and_outside_a_turn() -> None:
    from collections.abc import AsyncIterator

    from xr_ai_app._base.participant import CURRENT_TURN, Turn

    p, collector, runtime = await _bound()
    assert await p.reply("outside") == "outside"  # no turn: spoken as an update

    async def fragments() -> AsyncIterator[str]:
        yield "one "
        yield "two"

    CURRENT_TURN.set(Turn(text="q", timestamp_us=1, id="t1"))
    try:
        assert await p.reply(fragments()) == "one two"
        assert await p.reply("again") == "again"
    finally:
        CURRENT_TURN.set(None)
    assert [(kind, text) for _, kind, text in collector.seen if text] == [
        ("update", "outside"),
        ("reply", "one "),
        ("reply", "two"),
        ("reply", "again"),
    ]
    await p._scope.leave()
    assert await p.reply("late") == ""  # scope stopped admitting: nothing spoken
    await runtime.stop()


async def test_app_participants_lists_who_is_enrolled_now() -> None:
    from pathlib import Path

    from xr_ai_app._base.app import App, AppConfig

    app = App(AppConfig(config_dir=Path(".")))
    assert app.participants == []
    async with app.test_session() as s:
        await s.join("p1")
        await s.join("p2")
        assert [p.id for p in app.participants] == ["p1", "p2"]
        assert app.participants[0] is s.participant("p1")
        await s.leave("p1")
        assert [p.id for p in app.participants] == ["p2"]


async def test_release_forgets_sourced_utterances() -> None:
    p, _, runtime = await _bound()
    await p.say("x", source="a")
    assert p._sourced == {"a"}
    await p._release()
    assert p._sourced == set()
    await runtime.stop()
