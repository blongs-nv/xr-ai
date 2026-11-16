# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A virtual-time event loop: idle waits are skipped, real work still takes real time.

An app waits on its own timers far more than it computes: a camera step looks once a second, a timer counts down
ninety seconds, a video plays frames at its frame rate. :class:`VirtualTimeLoop` is asyncio's selector loop with one
change: when it would block and nothing can be ready except timers, it moves its clock to the next timer instead of
sleeping. Its clock is ``time.monotonic()`` plus how far it has skipped (``virtual_offset_s``).

It still waits for real when it might be waiting on something outside the program:

- real external work marked with ``xr_ai_next.clock.held`` (every real model call, a Conversation's streamed round, an
  A2A job), so their timeouts are not hit early and their latency counts as it would live;
- work running in a thread (``run_in_executor``, which ``asyncio.to_thread`` and file writes use);
- nothing scheduled at all (only I/O could wake it).

Frames stamped with ``xr_ai_next.clock.wall_us()`` age with this clock, so freshness checks see the same ages they
would live. Scripted roles return at once, so with them a window of minutes runs in well under a second.

Not held: other network I/O an app starts itself (an HTTP client of its own) - a virtual run can make its timeouts
fire early; run such pieces on the real clock. Trace event times (``t``, ``duration_s``) stay real.

Run a coroutine on it with :func:`run_virtual` (``asyncio.Runner`` with this loop); ``xr-ai-dev exercise`` and
``xr-ai-dev run`` use it with ``--clock virtual``.
"""

from __future__ import annotations

import asyncio
import contextlib
import selectors
import time
from collections.abc import Coroutine, Iterator
from typing import Any, TypeVar

__all__ = ["VirtualTimeLoop", "run_virtual"]

T = TypeVar("T")


class _SkippingSelector(selectors.DefaultSelector):  # type: ignore[misc,valid-type]
    """The loop's selector: polls first; when the loop is idle but for timers, skips to the timer."""

    loop: VirtualTimeLoop | None = None

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        events = super().select(0)
        if events or timeout == 0:
            return events
        loop = self.loop
        if loop is None or timeout is None or loop.held or loop.threads:
            return super().select(timeout)
        loop.virtual_offset_s += timeout
        loop.skipped += 1
        return []


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """asyncio's selector loop on a clock that skips idle waits (see the module docstring)."""

    def __init__(self) -> None:
        selector = _SkippingSelector()
        super().__init__(selector)
        selector.loop = self
        self.virtual_offset_s = 0.0
        """How far the clock has been moved ahead of ``time.monotonic()``."""
        self.held = 0
        """Real external work in flight (``xr_ai_next.clock.held``)."""
        self.threads = 0
        """Executor jobs in flight."""
        self.skipped = 0
        """How many waits were skipped."""

    def time(self) -> float:
        return time.monotonic() + self.virtual_offset_s

    @contextlib.contextmanager
    def hold(self) -> Iterator[None]:
        """Do not skip time while the block runs (what ``xr_ai_next.clock.held`` calls)."""
        self.held += 1
        try:
            yield
        finally:
            self.held -= 1

    def run_in_executor(self, executor: Any, func: Any, *args: Any) -> asyncio.Future[Any]:
        future = super().run_in_executor(executor, func, *args)
        self.threads += 1

        def done(_: asyncio.Future[Any]) -> None:
            self.threads -= 1

        future.add_done_callback(done)
        return future

    async def shutdown_default_executor(self, timeout: float | None = None) -> None:
        # Joins the executor's threads from a thread of its own, with a timeout on this clock: wait for real.
        with self.hold():
            await super().shutdown_default_executor(timeout)


def run_virtual(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion on a :class:`VirtualTimeLoop` (from synchronous code)."""
    with asyncio.Runner(loop_factory=VirtualTimeLoop) as runner:
        return runner.run(coro)
