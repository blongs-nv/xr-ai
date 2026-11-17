# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A countdown on the core clock, reporting its progress; no model calls."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from xr_ai_app import clock


@dataclass(frozen=True, slots=True)
class TimerState:
    """A sample of the countdown."""

    status: Literal["running", "completed"]
    duration_s: float
    elapsed_s: float
    remaining_s: float

    def summary(self) -> str:
        if self.status == "completed":
            return f"completed: {self.duration_s:g} seconds have passed."
        return f"running: {self.remaining_s:.0f} of {self.duration_s:g} seconds left."


async def timer(
    duration_s: float,
    *,
    interval_s: float = 1.0,
    on_state: Callable[[TimerState], Awaitable[None]] | None = None,
) -> TimerState:
    """Count down ``duration_s`` from now, awaiting ``on_state`` at most every ``interval_s``; return the final state.

    Time is read through ``xr_ai_app.clock``, so the devkit's virtual clock runs a three-minute steep in moments.
    The last sleep is bounded by the remaining time, so a long interval never delays completion.
    """
    if not (math.isfinite(duration_s) and math.isfinite(interval_s)) or duration_s <= 0 or interval_s <= 0:
        raise ValueError("duration_s and interval_s must be positive")
    started = clock.now()
    while True:
        elapsed = max(0.0, clock.now() - started)
        remaining = max(0.0, duration_s - elapsed)
        state = TimerState("completed" if remaining == 0 else "running", duration_s, elapsed, remaining)
        if on_state is not None:
            await on_state(state)
        if remaining == 0:
            return state
        await asyncio.sleep(min(interval_s, remaining))
