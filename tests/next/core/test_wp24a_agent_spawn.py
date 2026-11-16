# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Component.spawn`` is component-lifetime work owned by the app."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from xr_ai_next import Component
from xr_ai_next._base.app import App, AppConfig
from xr_ai_next._base.scope import ScopeTimeouts


def make_app() -> App:
    return App(AppConfig(config_dir=Path("."), timeouts=ScopeTimeouts(drain_s=0.2, hook_s=0.2, close_s=0.2)))


class Worker(Component):
    def __init__(self) -> None:
        super().__init__()
        self.log: list[str] = []

    async def blocked(self, tag: str) -> None:
        self.log.append(f"{tag}:start")
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.log.append(f"{tag}:cancelled")
            raise

    async def on_start(self) -> None:
        self.spawn(self.blocked("start"), key="start")


async def test_spawn_in_on_start_runs_until_stop_and_is_cancelled_by_it() -> None:
    app, worker = make_app(), Worker()
    app.add(worker)
    await app.start()
    await asyncio.sleep(0.01)
    assert worker.log == ["start:start"]
    await app.stop()
    assert worker.log == ["start:start", "start:cancelled"]


async def test_spawn_returns_a_task_and_keys_are_per_agent() -> None:
    app, a, b = make_app(), Worker(), Worker()
    app.add(a)
    app.add(b)
    await app.start()
    try:
        task = a.spawn(a.blocked("k"), key="k")
        assert isinstance(task, asyncio.Task)
        b.spawn(b.blocked("k"), key="k")  # same key on another agent is a different namespace
        coro = a.blocked("dup")
        with pytest.raises(ValueError, match="already running"):
            a.spawn(coro, key="k")
        a.spawn(a.blocked("second"), key="k", supersede=True)
        await asyncio.sleep(0.01)
        assert "k:cancelled" in a.log and "second:start" in a.log
        assert "k:cancelled" not in b.log
    finally:
        await app.stop()
    assert "second:cancelled" in a.log and "k:cancelled" in b.log


async def test_spawn_on_an_unregistered_agent_raises_runtime_error() -> None:
    worker = Worker()
    coro = worker.blocked("x")
    with pytest.raises(RuntimeError, match="not added to an app"):
        worker.spawn(coro)


async def test_spawn_while_stopping_is_refused_and_a_failing_task_is_contained() -> None:
    app, worker = make_app(), Worker()
    app.add(worker)
    await app.start()

    async def boom() -> None:
        raise ValueError("contained")

    task = worker.spawn(boom())
    await asyncio.wait({task})
    assert isinstance(task.exception(), ValueError)  # logged and reported; only an awaiting caller sees it (R4.6)
    await app.stop()
    coro = worker.blocked("late")
    with pytest.raises(RuntimeError, match="stopping"):
        worker.spawn(coro)
    coro.close()
