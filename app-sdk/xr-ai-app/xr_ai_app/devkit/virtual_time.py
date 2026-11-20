# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A virtual-time event loop: idle waits are skipped, real work still takes real time, or a declared time.

An app waits on its own timers far more than it computes: a camera step looks once a second, a timer counts down
ninety seconds, a video plays frames at its frame rate. :class:`VirtualTimeLoop` is asyncio's selector loop with one
change: when it would block and nothing can be ready except timers, it moves its clock to the next timer instead of
sleeping.

It still waits for real when it might be waiting on something outside the program:

- real external work marked with ``xr_ai_app.clock.held`` (every real model call, a Conversation's streamed round, an
  A2A job), so their timeouts are not hit early;
- work running in a thread (``run_in_executor``, which ``asyncio.to_thread`` and file writes use);
- nothing scheduled at all (only I/O could wake it).

How much a model call moves the clock is the loop's choice:

- **Without a latency profile** (the default), it moves with real time while it waits, so a call's latency counts as
  it took on this machine. The clock is ``time.monotonic()`` plus how far it has skipped (``virtual_offset_s``).
- **With a** :class:`LatencyProfile`, the clock moves only by skipping: each model call ends at its start plus the
  time the profile declares for its role, however long it took here, and timers due before then (the next camera
  look) still fire while it runs. The same profile gives the same timeline on any machine. Work that is not a role's
  call (unlabelled) costs no time. A call that has not returned ``stall_s`` (real) after it started lets the clock
  move with real time again, so its own timeout can fire.

Either way the loop records how long each role's calls took for real (:attr:`VirtualTimeLoop.real_latencies_s`), so
a run on the target machine can be turned into its profile.

Frames stamped with ``xr_ai_app.clock.wall_us()`` age with this clock, so freshness checks see the same ages they
would live. Scripted roles return at once, so with them a window of minutes runs in well under a second.

Not held: other network I/O an app starts itself (an HTTP client of its own) - a virtual run can make its timeouts
fire early; run such pieces on the real clock. Trace event times (``t``, ``duration_s``) stay real.

Run a coroutine on it with :func:`run_virtual` (``asyncio.Runner`` with this loop); ``xr-ai-dev exercise`` and
``xr-ai-dev run`` use it with ``--clock virtual``.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import selectors
import time
from collections.abc import Coroutine, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

__all__ = ["LatencyProfile", "VirtualTimeLoop", "run_virtual"]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """How long each role's model call takes on a virtual clock, in seconds, by role name.

    ``default`` is for a role the profile does not name (``None``: such a call is an error, so a profile cannot miss
    a role silently); ``scale`` multiplies every value (2.0: a machine twice as slow). Load one from YAML with
    :meth:`load` (``perceive: 1.1``, ``default: 0.5``); :meth:`zero` makes every call instant.
    """

    seconds: Mapping[str, float] = field(default_factory=dict)
    default: float | None = None
    scale: float = 1.0

    @classmethod
    def zero(cls) -> LatencyProfile:
        """Every model call instant: what a step does, with time standing still while models answer."""
        return cls(default=0.0)

    @classmethod
    def load(cls, path: Path | str, *, scale: float = 1.0) -> LatencyProfile:
        """A profile from YAML: ``{role: seconds, ...}``, optionally ``default: seconds``."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict) or not all(isinstance(v, int | float) for v in data.values()):
            raise ValueError(f"{path}: a latency profile maps role names (and 'default') to seconds")
        default = data.pop("default", None)
        return cls({str(k): float(v) for k, v in data.items()}, default, scale)

    def seconds_for(self, role: str | None) -> float:
        """The declared time of one call of ``role`` (unlabelled work: none)."""
        if role is None:
            return 0.0
        if role in self.seconds:
            return self.seconds[role] * self.scale
        if self.default is None:
            raise KeyError(f"the latency profile names no role {role!r}; add it, or a default")
        return self.default * self.scale


@dataclass(slots=True)
class HeldCall:
    """One piece of real external work in flight, as a virtual loop tracks it."""

    role: str | None
    started_real_s: float
    done_at_s: float | None = None
    """Under a latency profile: when it ends on the virtual clock (its start plus its declared time)."""


class _SkippingSelector(selectors.DefaultSelector):  # type: ignore[misc,valid-type]
    """The loop's selector: polls first; when the loop is idle but for timers, skips to the timer."""

    loop: VirtualTimeLoop | None = None

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        events = super().select(0)
        if events or timeout == 0:
            return events
        loop = self.loop
        if loop is None or timeout is None:
            return super().select(timeout)
        real_wait_s = loop.skip_or_wait(timeout)
        if real_wait_s is None:
            return []  # skipped to the next timer
        started = time.monotonic()
        events = super().select(real_wait_s)
        loop.waited_real(time.monotonic() - started)
        return events


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """asyncio's selector loop on a clock that skips idle waits (see the module docstring)."""

    def __init__(self, latency: LatencyProfile | None = None, *, stall_s: float = 60.0) -> None:
        selector = _SkippingSelector()
        super().__init__(selector)
        selector.loop = self
        self.latency = latency
        """``None``: a model call moves the clock as it took here; a profile: as it declares."""
        self.stall_s = stall_s
        self._skipped_s = 0.0  # without a profile: how far the clock is ahead of time.monotonic()
        # With a profile: the clock, moved only by skipping. It starts at 0, where a float's steps are fine enough
        # that a time left (end - now) always moves it.
        self._now = 0.0
        self._started_monotonic = time.monotonic()
        self.held_calls: list[HeldCall] = []
        """Real external work in flight (``xr_ai_app.clock.held``)."""
        self.threads = 0
        """Executor jobs in flight."""
        self.skipped = 0
        """How many waits were skipped."""
        self.real_latencies_s: dict[str, list[float]] = {}
        """How long each role's calls took for real, by role name, in the order they ended."""

    @property
    def held(self) -> int:
        """How many pieces of real external work are in flight."""
        return len(self.held_calls)

    @property
    def virtual_offset_s(self) -> float:
        """How far the clock is ahead of ``time.monotonic()`` (what ``xr_ai_app.clock.wall_us`` adds)."""
        if self.latency is None:
            return self._skipped_s
        return self._now - (time.monotonic() - self._started_monotonic)  # wall time starts at the real one

    def time(self) -> float:
        return time.monotonic() + self._skipped_s if self.latency is None else self._now

    def skip_or_wait(self, timeout: float) -> float | None:
        """The next timer is ``timeout`` away and nothing is ready: skip to it (``None``), or wait for real (the
        seconds to block)."""
        if self.latency is None:
            if self.held_calls or self.threads:
                return timeout
            self._skipped_s += timeout
            self.skipped += 1
            return None
        next_timer = self._now + timeout
        waiting = [c.done_at_s for c in self.held_calls if c.done_at_s is not None]
        if self.threads or (waiting and min(waiting) <= next_timer):
            return timeout  # a call due before the next timer, or a thread: its result is needed first
        self._now = max(next_timer, math.nextafter(self._now, math.inf))  # always forward, if only by a float step
        self.skipped += 1
        return None

    def waited_real(self, elapsed_s: float) -> None:
        """After a real wait: under a profile the clock stays put, unless a call has stalled (then real time
        counts, so its own timeout can fire)."""
        if self.latency is None:
            return
        now_real = time.monotonic()
        if any(now_real - c.started_real_s > self.stall_s for c in self.held_calls):
            self._now += elapsed_s

    @contextlib.contextmanager
    def hold(self, role: str | None = None) -> Iterator[HeldCall]:
        """Real external work for the block's duration (what ``xr_ai_app.clock.held`` calls): never skipped past.
        Under a profile the call is due at its start plus its declared time (``done_at_s``)."""
        call = HeldCall(role, time.monotonic())
        if self.latency is not None:
            call.done_at_s = self._now + self.latency.seconds_for(role)
        self.held_calls.append(call)
        try:
            yield call
        finally:
            self.held_calls.remove(call)
            if role is not None:
                self.real_latencies_s.setdefault(role, []).append(time.monotonic() - call.started_real_s)
            if call.done_at_s is not None and call.done_at_s < self._now:
                call.done_at_s = None  # a stalled call ran past its declared end: it ends now

    def call_at(self, when: float, callback: Any, *args: Any, context: Any = None) -> asyncio.TimerHandle:
        """Under a profile, a timer moves the clock to its time as it fires. asyncio fires a timer due within its
        clock's resolution as if due now; with a clock that moves only by skipping, it would fire before the clock
        reached it, and a wait for the time left (end - now: tiny, but positive) would repeat forever."""
        if self.latency is None:
            return super().call_at(when, callback, *args, context=context)

        def at_its_time(*callback_args: Any) -> None:
            if when >= self._now:  # due now or later: the clock moves to it, if only by a float step
                self._now = max(when, math.nextafter(self._now, math.inf))
            callback(*callback_args)

        return super().call_at(when, at_its_time, *args, context=context)

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


def run_virtual(coro: Coroutine[Any, Any, T], *, latency: LatencyProfile | None = None) -> T:
    """Run ``coro`` to completion on a :class:`VirtualTimeLoop` (from synchronous code), with ``latency`` (``None``:
    model calls take the time they take here)."""
    with asyncio.Runner(loop_factory=lambda: VirtualTimeLoop(latency)) as runner:
        return runner.run(coro)
