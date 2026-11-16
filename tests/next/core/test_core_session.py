# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The App on the test kit: ``app.test_session()`` with FakeHub and FakeModels."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from xr_ai_next import Component
from xr_ai_next._base import testing
from xr_ai_next._base.app import App, AppConfig, VoiceConfig
from xr_ai_next._base.participant import Participant, Turn
from xr_ai_next._base.scope import ScopeTimeouts
from xr_ai_next._base.testing import FakeModels
from xr_ai_runtime import RuntimeClosedError


def make_app() -> App:
    return App(
        AppConfig(
            config_dir=Path("."),
            voice=VoiceConfig(enabled=False),
            timeouts=ScopeTimeouts(drain_s=0.2, hook_s=0.2, close_s=0.2),
        )
    )


async def test_a_typed_query_runs_a_turn_and_its_reply_is_recorded_as_one_message() -> None:
    app = make_app()

    @app.on_turn
    async def handle(turn: Turn, p: Participant) -> None:
        async def fragments() -> AsyncIterator[str]:
            yield "you said "
            yield turn.text

        await p.reply(fragments())

    async with app.test_session() as s:
        await s.join()
        await s.say("hello")
        await s.settle()
        assert s.spoken() == ["you said hello"]


async def test_say_records_updates_with_their_urgency() -> None:
    app = make_app()

    class Greeter(Component):
        async def on_participant_enrolled(self, p: Participant) -> None:
            await p.say("welcome")
            await p.say("careful", urgent=True)

    app.add(Greeter())
    async with app.test_session() as s:
        await s.join()
        assert s.spoken() == ["welcome", "careful"]
        assert s.hub.urgent() == [False, True]


async def test_a_failing_turn_reports_error_status_on_the_hub_and_the_session_survives() -> None:
    app = make_app()

    @app.on_turn
    async def handle(turn: Turn, p: Participant) -> None:
        if turn.text == "boom":
            raise RuntimeError("bad turn")
        await p.say("ok")

    async with app.test_session() as s:
        await s.join()
        await s.say("boom")
        await s.settle()
        assert [t for t, _ in s.hub.statuses()] == ["error"]
        await s.say("fine")
        await s.settle()
        assert s.spoken() == ["ok"]


async def test_settle_waits_for_a_turn_blocked_on_a_model_call() -> None:
    app = make_app()
    models = FakeModels({"answer": ["the answer"]})

    @app.on_turn
    async def handle(turn: Turn, p: Participant) -> None:
        await p.reply(await app.role("answer").ask(None, turn.text))

    async with testing.test_session(app, models=models) as s:
        await s.join()
        await s.say("q")
        await s.settle()
        assert s.spoken() == ["the answer"]
        assert [c.prompt for c in models.calls("answer")] == ["q"]


async def test_after_stop_the_hub_is_closed_and_no_task_can_publish() -> None:
    app = make_app()
    async with app.test_session() as s:
        await s.join()
        p = app._participants["p1"]
        hub = s.hub
    assert hub.closed
    await p.say("late")  # silent no-op, never RuntimeClosedError
    with pytest.raises(RuntimeClosedError):
        hub.publish_output("p1", "direct")


async def test_a_second_session_cannot_install_doubles_after_start() -> None:
    app = make_app()
    async with app.test_session() as s:
        with pytest.raises(RuntimeError, match="before app.start"):
            app._install_test_doubles(s.hub, s.models)


async def test_is_idle_ignores_parked_and_long_lived_tasks_and_sees_runnable_turns() -> None:
    app = make_app()
    async with app.test_session() as s:
        await s.join()
        p = app._participants["p1"]
        p.spawn(asyncio.sleep(10), key="turn")
        p.spawn(_busy(), key="loop")
        await asyncio.sleep(0.01)
        assert app._is_idle() is True

        async def runnable() -> None:
            while True:
                await asyncio.sleep(0)

        p.spawn(runnable())  # un-keyed and runnable
        await asyncio.sleep(0.01)
        assert app._is_idle() is False


async def _busy() -> None:
    while True:
        await asyncio.sleep(0)
