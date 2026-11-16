# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Timer``: a step that counts down, with no model calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from xr_ai_app import clock

from .perception import Done, Report

__all__ = ["Timer"]


class Timer:
    """Count down ``seconds``, reporting the time left every ``interval_s``; ends in ``Done(seconds)``.

    The running event loop's clock (monotonic on a normal loop, so a wall-clock change cannot change the duration;
    skipped ahead on the devkit's virtual-time loop). ``now`` and ``sleep`` override it.
    """

    def __init__(
        self,
        seconds: float,
        *,
        interval_s: float = 5.0,
        now: Callable[[], float] = clock.now,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        if seconds <= 0 or interval_s <= 0:
            raise ValueError("seconds and interval_s must be positive")
        self.seconds = seconds
        self.interval_s = interval_s
        self._clock = now
        self._sleep = sleep

    async def run(self, report: Report) -> Done:
        started = self._clock()
        while True:
            left = max(0.0, self.seconds - (self._clock() - started))
            await report(f"{left:.0f} seconds left" if left else "time is up")
            if left == 0:
                return Done(self.seconds)
            await self._sleep(min(self.interval_s, left))

    def __repr__(self) -> str:
        return f"Timer({self.seconds:g})"
