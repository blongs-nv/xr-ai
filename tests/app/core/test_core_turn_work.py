# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Work done inside a turn: a newer turn cancels it, plain asyncio concurrency is safe, failures reach the awaiter.

Rules R1.9 and R4.6 of the lifecycle contract.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from xr_ai_app._base.app import App, AppConfig
from xr_ai_app._base.participant import Participant, Turn
from xr_ai_app._base.scope import install_relay_task_factory, uninstall_relay_task_factory


def _app() -> App:
    return App(AppConfig(config_dir=Path(".")))


async def _until(predicate, timeout_s: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout_s)


async def test_a_newer_turn_cancels_the_work_awaited_by_the_previous_turn() -> None:
    app = _app()
    cancelled: list[str] = []
    started: list[str] = []

    async def research(p: Participant, topic: str) -> str:
        started.append(topic)
        try:
            await p.say(f"Looking into {topic}.")
            if topic == "first":
                await asyncio.Event().wait()  # never finishes on its own
            return f"{topic} done"
        except asyncio.CancelledError:
            cancelled.append(topic)
            raise

    @app.on_turn
    async def answer(turn: Turn, p: Participant) -> None:
        await p.reply(await research(p, turn.text))

    async with app.test_session() as s:
        await s.join()
        await s.say("first")
        await _until(lambda: started == ["first"])
        await s.say("second")
        await _until(lambda: "second done" in s.spoken())
        assert cancelled == ["first"]
        assert "first done" not in s.spoken()


async def test_task_group_sub_tasks_may_speak_concurrently_and_are_cancelled_with_their_turn() -> None:
    """Sibling tasks from a TaskGroup each get their own Relay scope stack, so concurrent output does not fail."""
    app = _app()
    cancelled: list[str] = []
    started: list[str] = []
    errors: list[str] = []

    async def part(p: Participant, name: str, block: bool) -> str:
        started.append(name)
        try:
            await p.say(f"Looking into {name}.")
            await asyncio.sleep(0)
            await p.say(f"Still on {name}.")
            if block:
                await asyncio.Event().wait()
            return name
        except asyncio.CancelledError:
            cancelled.append(name)
            raise

    @app.on_turn
    async def answer(turn: Turn, p: Participant) -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                a = tg.create_task(part(p, f"{turn.text}/a", turn.text == "first"))
                b = tg.create_task(part(p, f"{turn.text}/b", turn.text == "first"))
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion below
            errors.append(repr(exc))
            raise
        await p.reply(f"{a.result()} and {b.result()}")

    async with app.test_session() as s:
        await s.join()
        await s.say("first")
        await _until(lambda: len(started) == 2)
        await s.say("second")
        await _until(lambda: "second/a and second/b" in s.spoken())
    assert errors == []
    assert sorted(cancelled) == ["first/a", "first/b"]


async def test_awaiting_a_spawned_job_inside_a_turn_raises_its_failure() -> None:
    app = _app()
    seen: list[str] = []

    async def job() -> str:
        raise LookupError("no such instrument")

    @app.on_turn
    async def answer(turn: Turn, p: Participant) -> None:
        try:
            await p.spawn(job(), key="lookup")
        except LookupError as exc:
            seen.append(str(exc))
            await p.reply("I could not find it.")

    async with app.test_session() as s:
        await s.join()
        await s.say("where is the scale?")
        await _until(lambda: "I could not find it." in s.spoken())
    assert seen == ["no such instrument"]


async def test_the_task_factory_is_installed_while_the_app_runs_and_restored_after() -> None:
    loop = asyncio.get_running_loop()
    before = loop.get_task_factory()
    app = _app()
    async with app.test_session():
        assert loop.get_task_factory() is not before
    assert loop.get_task_factory() is before


async def test_the_task_factory_is_counted_per_loop_and_chains_an_earlier_factory() -> None:
    loop = asyncio.get_running_loop()
    before = loop.get_task_factory()
    made: list[str] = []

    def earlier(loop: asyncio.AbstractEventLoop, coro, **kwargs):  # type: ignore[no-untyped-def]
        made.append("earlier")
        return asyncio.Task(coro, loop=loop, **kwargs)

    loop.set_task_factory(earlier)
    try:
        install_relay_task_factory(loop)
        install_relay_task_factory(loop)
        ours = loop.get_task_factory()
        assert ours is not earlier
        await asyncio.create_task(asyncio.sleep(0))
        assert made == ["earlier"]  # the earlier factory still creates the task
        uninstall_relay_task_factory(loop)
        assert loop.get_task_factory() is ours  # a second app still runs
        uninstall_relay_task_factory(loop)
        assert loop.get_task_factory() is earlier
    finally:
        loop.set_task_factory(before)


@pytest.mark.parametrize("explicit", [True, False])
async def test_only_tasks_without_an_explicit_context_are_forked(explicit: bool) -> None:
    import contextvars  # noqa: PLC0415

    loop = asyncio.get_running_loop()
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")
    install_relay_task_factory(loop)
    try:
        context = contextvars.copy_context()
        context.run(marker.set, "given")

        async def read() -> str:
            return marker.get()

        task = loop.create_task(read(), context=context) if explicit else loop.create_task(read())
        assert await task == ("given" if explicit else "unset")
    finally:
        uninstall_relay_task_factory(loop)


async def test_forking_leaves_the_creators_context_without_a_relay_stack() -> None:
    """Relay creates a stack in the context that forks; the core forks in a copy, so the creator's stays untouched."""
    import contextvars  # noqa: PLC0415

    def relay_vars() -> set[str]:
        return {var.name for var in contextvars.copy_context() if var.name == "scope_stack"}

    loop = asyncio.get_running_loop()

    def create() -> set[str]:
        install_relay_task_factory(loop)
        try:
            loop.create_task(asyncio.sleep(0))
        finally:
            uninstall_relay_task_factory(loop)
        return relay_vars()

    assert contextvars.Context().run(create) == set()
