# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A small, correct implementation of the lifecycle contract used only to test the contract suite.

It stands in for the real ``App`` through the same seam. ``mutations`` switches on one
deliberate violation each, so tests can prove the suite notices it.
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any

from xr_ai_app._base.app import AppConfig
from xr_ai_app._base.scope import ScopeTimeouts
from xr_ai_app._base.testing import FakeHub, FakeModels


class RefParticipant:
    def __init__(self, app: RefApp, pid: str) -> None:
        self.app = app
        self.id = pid
        self.state: dict[str, Any] = {}
        self.admitting = True
        self.tasks: dict[str, asyncio.Task[Any]] = {}
        self._anon = 0

    def spawn(self, coro: Any, *, key: str | None = None, supersede: bool = False) -> asyncio.Task[Any]:
        if not self.admitting and "admit_after_leave" not in self.app.mutations:
            coro.close()
            raise RuntimeError("participant scope is not admitting")
        old = self.tasks.get(key) if key else None
        if old is not None and not old.done():
            if not supersede:
                coro.close()
                raise ValueError(f"task {key!r} is live")
            old.cancel()
        else:
            old = None
        if key is None:
            self._anon += 1
        slot = key if key is not None else f"_anon{self._anon}"

        async def run() -> Any:
            if old is not None:
                await asyncio.gather(old, return_exceptions=True)
            try:
                return await coro
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if "swallow_errors" not in self.app.mutations:
                    self.app.hub.publish_status(self.id, "error", task=slot, error=repr(exc))
                raise

        task = asyncio.get_running_loop().create_task(run())
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        self.tasks[slot] = task
        return task

    async def cancel(self, key: str) -> bool:
        task = self.tasks.get(key)
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def say(self, text: str, *, urgent: bool = False) -> None:
        if self.admitting and not self.app.hub.closed:
            self.app.hub.publish_output(self.id, text, urgent=urgent)

    async def reply(self, text: Any) -> None:
        await self.say(text if isinstance(text, str) else "".join([c async for c in text]))

    async def status(self, text: str, **detail: Any) -> None:
        if self.admitting and not self.app.hub.closed:
            self.app.hub.publish_status(self.id, text, **detail)

    async def leave(self) -> None:
        t = self.app._config.timeouts
        if "admit_after_leave" not in self.app.mutations:
            self.admitting = False
        live = [x for x in self.tasks.values() if not x.done()]
        for task in live:
            task.cancel()
        _, pending = await asyncio.wait(live, timeout=t.drain_s) if live else (set(), set())
        if pending:
            self.app.hub.publish_status(self.id, "error", abandoned=len(pending))
        hooks = self.app.agents if "no_reverse" in self.app.mutations else list(reversed(self.app.agents))
        for agent in hooks:
            try:
                await asyncio.wait_for(agent.on_participant_unenrolled(self), t.hook_s)
            except Exception:  # noqa: BLE001 - contains hook failures (R2.7)
                self.app.hook_errors += 1
        self.admitting = False


class RefApp:
    def __init__(self, *, mutations: frozenset[str] = frozenset(), drain_s: float = 0.2) -> None:
        self.mutations = mutations
        self._config = AppConfig(
            config_dir=Path("."),
            timeouts=ScopeTimeouts(drain_s=drain_s, hook_s=0.5, close_s=0.5),
        )
        self.agents: list[Any] = []
        self.participants: dict[str, RefParticipant] = {}
        self.hub: FakeHub = None  # type: ignore[assignment]
        self.models = FakeModels()
        self.hook_errors = 0
        self._started: list[Any] = []
        self._stop_task: asyncio.Task[None] | None = None

    def add(self, agent: Any) -> Any:
        self.agents.append(agent)
        return agent

    def _install_test_doubles(self, hub: FakeHub, models: FakeModels) -> None:
        self.hub, self.models = hub, models
        hub.attach(self)

    def role(self, name: str) -> Any:
        return self.models.role(name)

    async def start(self) -> None:
        if "signals" in self.mutations:
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, lambda: None)
        try:
            for agent in self.agents:
                self._started.append(agent)
                await agent.on_start()
        except Exception:
            self._started.pop()
            await self.stop()
            raise

    async def stop(self) -> None:
        if self._stop_task is None:
            self._stop_task = asyncio.get_running_loop().create_task(self._stop())
        await asyncio.shield(self._stop_task)

    async def _stop(self) -> None:
        self.hub.detach(self)
        if "leaky_stop" not in self.mutations:
            await asyncio.gather(*(p.leave() for p in self.participants.values()))
            self.participants.clear()
        for agent in reversed(self._started):
            try:
                await asyncio.wait_for(agent.on_stop(), self._config.timeouts.hook_s)
            except Exception:  # noqa: BLE001 - R2.7 applies to stop hooks too
                self.hook_errors += 1
        self.hub.close()

    # hub listener
    async def on_hub_join(self, pid: str) -> None:
        p = self.participants[pid] = RefParticipant(self, pid)
        for agent in self.agents:
            await agent.on_participant_enrolled(p)

    async def on_hub_leave(self, pid: str) -> None:
        await self.participants.pop(pid).leave()

    async def on_hub_frame(self, pid: str, frame: Any) -> None:
        return None

    async def on_hub_query(self, pid: str, text: str, timestamp_us: int) -> None:
        return None
