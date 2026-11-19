# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hooks for a virtual-time event loop (private). On a normal event loop every one of them is a no-op.

A virtual-time loop (``xr_ai_app.devkit.VirtualTimeLoop``) skips ahead when the program is only waiting on its own
timers (a step's look interval, a timer, a video's frame spacing), so a test of a minutes-long window runs in
moments. Two things it must know about, and these hooks tell it:

- **The wall clock.** Frame freshness is "wall time now minus the frame's capture time". A virtual loop keeps a
  ``virtual_offset_s`` (how far it has skipped); :func:`wall_us` adds it, so frames stamped with it age the way the
  loop's clock moves. On a normal loop the offset is absent and this is ``time.time_ns() // 1000``.
- **Real external work.** A model call or an HTTP request takes real time and its timeouts run on the loop's clock;
  skipping ahead while one is in flight would fire those timeouts early. :func:`held` (and :func:`held_call`,
  :func:`held_role_call`, :class:`HeldStream`) mark such work; a virtual loop does not skip while any is held. A
  role's calls are labelled with the role's name, so a loop with a latency profile can charge each one the time the
  profile declares for that role instead of the time it took here (``xr_ai_app.devkit.LatencyProfile``).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, TypeVar

__all__ = ["HeldStream", "held", "held_call", "held_role_call", "wall_us"]

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def _loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def wall_us() -> int:
    """Wall-clock now in microseconds, moved by the running loop's virtual offset (none on a normal loop)."""
    offset = getattr(_loop(), "virtual_offset_s", 0.0)
    return time.time_ns() // 1000 + int(offset * 1_000_000)


@contextlib.contextmanager
def held(label: str | None = None) -> Iterator[Any]:
    """Mark real external work for the duration of the block (a no-op on a normal loop). ``label`` names it (a
    role's name). Yields the virtual loop's record of the call (``None`` on a normal loop); a block that is not
    awaited after costs no virtual time under a latency profile: use :func:`held_call` or :func:`held_role_call`."""
    hold = getattr(_loop(), "hold", None)
    if hold is None:
        yield None
        return
    with hold(label) as call:
        yield call


async def _until_declared_end(call: Any) -> None:
    """Under a latency profile, wait (on the virtual clock) until the call's declared end; otherwise nothing."""
    done_at_s = getattr(call, "done_at_s", None)
    if done_at_s is not None:
        await asyncio.sleep(max(0.0, done_at_s - asyncio.get_running_loop().time()))


def held_call(fn: F) -> F:
    """Decorate an async function that does real external work (unlabelled: a profile charges it nothing)."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        with held() as call:
            result = await fn(*args, **kwargs)
        await _until_declared_end(call)
        return result

    return wrapper  # type: ignore[return-value]


def held_role_call(fn: F) -> F:
    """Decorate a role's async method that calls its model: held, labelled with the role's ``name``."""

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with held(self.name) as call:
            result = await fn(self, *args, **kwargs)
        await _until_declared_end(call)
        return result

    return wrapper  # type: ignore[return-value]


class HeldStream:
    """An async iterator of real external work: held from its first item until it ends, fails or is closed.
    ``label`` names it (a role's name), as :func:`held`."""

    def __init__(self, inner: AsyncIterator[Any], *, label: str | None = None) -> None:
        self._inner = inner
        self._label = label
        self._hold: contextlib.ExitStack | None = None
        self._call: Any = None

    def __aiter__(self) -> HeldStream:
        return self

    async def __anext__(self) -> Any:
        if self._hold is None:
            self._hold = contextlib.ExitStack()
            self._call = self._hold.enter_context(held(self._label))
        try:
            return await self._inner.__anext__()
        except StopAsyncIteration:
            call = self._call
            self._release()
            await _until_declared_end(call)  # under a latency profile, the stream ends at its declared time
            raise
        except BaseException:
            self._release()
            raise

    async def aclose(self) -> None:
        self._release()
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

    def _release(self) -> None:
        if self._hold is not None:
            self._hold.close()
            self._hold = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
