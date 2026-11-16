# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``TestSession`` and ``test_session``: run an app on a :class:`FakeHub` and :class:`FakeModels`."""

from __future__ import annotations

import asyncio
import time
import warnings
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from ..frames import Frame
from ._hub import FakeHub
from ._models import FakeModels

if TYPE_CHECKING:
    from ..app import App
    from ..participant import Participant


SETTLE_TIMEOUT_S = 5.0
"""Seconds :meth:`TestSession.settle` waits before raising ``TimeoutError``."""


class TestSession:
    """Drives an app on CPU: join, speak, show frames, leave."""

    __test__ = False

    def __init__(self, app: App, hub: FakeHub, models: FakeModels) -> None:
        self._app = app
        self.hub = hub
        """The fake hub the app is attached to, for inspecting statuses and events."""

        self.models = models
        """The scripted models the app resolves its roles from."""

    @property
    def app(self) -> App:
        """The app under test."""
        return self._app

    async def join(self, participant_id: str = "p1") -> None:
        await self.hub.join(participant_id)

    async def say(self, text: str, *, participant_id: str = "p1") -> None:
        await self.hub.text_query(text, participant_id)

    async def show_frame(self, frame: Frame, *, participant_id: str = "p1") -> None:
        await self.hub.push_frame(frame, participant_id)

    async def leave(self, participant_id: str = "p1") -> None:
        await self.hub.leave(participant_id)

    def participant(self, participant_id: str = "p1") -> Participant:
        """Return the joined participant's :class:`~.participant.Participant`, to drive its tasks or inspect its state.

        Raises ``KeyError`` when the participant has not joined (or has left).
        """
        p = self._app._participants.get(participant_id)
        if p is None:
            raise KeyError(f"participant {participant_id!r} is not joined")
        return p

    def spoken(self, participant_id: str = "p1") -> list[str]:
        """Return everything the app said to the participant, in order."""
        return self.hub.spoken(participant_id)

    async def settle(self) -> None:
        """Wait until every scope is idle.

        Idle means: no fake model call is in flight and ``app._is_idle()`` (a
        warning is issued when the app lacks it) is true,
        both holding across three event-loop turns. Raises ``TimeoutError``.
        """
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        stable = 0
        is_idle = getattr(self._app, "_is_idle", None)
        if is_idle is None:
            warnings.warn(
                "App has no _is_idle(); settle() only waits for fake model calls",
                stacklevel=2,
            )
        while stable < 3:
            await asyncio.sleep(0)
            idle = not self.models.any_active() and (is_idle() if is_idle is not None else True)
            if idle:
                stable += 1
                continue
            stable = 0
            if time.monotonic() > deadline:
                raise TimeoutError("session did not settle")
            await asyncio.sleep(0.005)


@asynccontextmanager
async def _test_session(
    app: App, models: FakeModels | None, replace: Mapping[type, Any] | None
) -> AsyncIterator[TestSession]:
    from ..hub import Hub  # noqa: PLC0415 - keeps the test kit's import light

    rest = dict(replace or {})
    hub = rest.pop(Hub, None) or FakeHub()
    if not isinstance(hub, FakeHub):
        raise TypeError(f"replace[Hub] must be a FakeHub, got {type(hub).__name__}")
    models = models if models is not None else FakeModels()
    install = getattr(app, "_install_test_doubles", None)
    if install is None:
        raise NotImplementedError("App has no _install_test_doubles seam")
    install(hub, models, replace=rest) if rest else install(hub, models)
    from .._tracing import MemorySink  # noqa: PLC0415

    trace = MemorySink()
    tracer = getattr(app, "trace", None)  # an app stand-in (the contract suite's reference app) may have none
    if tracer is not None:
        tracer.add(trace)
    await app.start()
    try:
        session = TestSession(app, hub, models)
        session.trace = trace  # every trace event of the session
        yield session
    finally:
        await app.stop()
        hub.close()
        if tracer is not None:
            tracer.remove(trace)


def test_session(app: App, *, models: FakeModels | None = None, replace: Mapping[type, Any] | None = None) -> Any:
    """Return an async context manager that starts ``app`` on a :class:`FakeHub` and yields a session.

    ``replace`` swaps added components by class before start: ``replace={Hub: FakeHub()}`` puts your own
    :class:`FakeHub` in the place of the app's ``Hub``. Without a ``Hub`` key a fresh
    ``FakeHub`` is used, and it is added after the app's own components when it has no ``Hub``. That is test-kit
    convenience, so the many tests of an app that wires nothing still have a device side, not an SDK default. An
    app without an enrollment policy (a hand-built ``App(config)``) gets ``PerConnection(admit=None)`` for the
    session; ``App.from_config`` always carries its own.

    The app is attached to the fake hub and ``models`` (a fresh empty :class:`FakeModels` by default)
    through ``App._install_test_doubles(hub, models)``; it is started on entry and stopped on exit,
    after which the hub is closed so a task that outlived the stop raises ``RuntimeClosedError``.
    No signal handler is installed.
    """
    return _test_session(app, models, replace)


test_session.__test__ = False  # type: ignore[attr-defined]
