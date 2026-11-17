# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A wait step: a countdown on the core clock (:func:`~.timer.timer`); no camera, no models."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..step_api import StepIO
from .timer import TimerState, timer


@dataclass(frozen=True, slots=True)
class WaitStep:
    """Wait ``duration_s`` seconds, reporting the time left."""

    id: str
    duration_s: float
    say: str | None = None
    interval_s: float = 1.0

    def lasting(self, duration_s: float) -> WaitStep:
        """A copy that waits ``duration_s`` (a duration known only at run time)."""
        return replace(self, duration_s=duration_s)

    def format(self, **values: object) -> WaitStep:
        """A copy with ``{name}`` placeholders filled in the spoken line."""
        return replace(self, say=self.say.format(**values) if self.say else self.say)

    async def run(self, io: StepIO) -> TimerState:
        return await timer(self.duration_s, interval_s=self.interval_s, on_state=io.report(self.id))
