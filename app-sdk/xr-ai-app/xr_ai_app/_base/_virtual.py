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
  :class:`HeldStream`) mark such work; a virtual loop does not skip while any is held.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, TypeVar

__all__ = ["HeldStream", "held", "held_call", "wall_us"]

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
def held() -> Iterator[None]:
    """Mark real external work for the duration of the block (a no-op on a normal loop)."""
    hold = getattr(_loop(), "hold", None)
    if hold is None:
        yield
        return
    with hold():
        yield


def held_call(fn: F) -> F:
    """Decorate an async function that does real external work (a model call)."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        with held():
            return await fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


class HeldStream:
    """An async iterator of real external work: held from its first item until it ends, fails or is closed."""

    def __init__(self, inner: AsyncIterator[Any]) -> None:
        self._inner = inner
        self._hold: contextlib.ExitStack | None = None

    def __aiter__(self) -> HeldStream:
        return self

    async def __anext__(self) -> Any:
        if self._hold is None:
            self._hold = contextlib.ExitStack()
            self._hold.enter_context(held())
        try:
            return await self._inner.__anext__()
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
