# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A wait step: a countdown on the core clock (:func:`~.timer.timer`); no camera, no models.

Its definition has no section of its own: ``say``, and ``done`` filled with the build-time values
(:meth:`WaitStep.format`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from ...steps.step_api import StepIO
from ...steps.step_definitions import StepDefinition
from .timer import TimerState, timer


@dataclass(frozen=True, slots=True)
class WaitStep:
    """Wait ``duration_s`` seconds, reporting the time left."""

    scoring: ClassVar[str] = f"{__package__}.wait_scoring"

    id: str
    duration_s: float
    say: str | None = None
    done: str | None = None
    values: Mapping[str, object] = field(default_factory=dict)
    interval_s: float = 1.0

    @classmethod
    def define(cls, definition: StepDefinition, duration_s: float = 0.0, **code: Any) -> WaitStep:
        """The step from its definition (its duration is usually set later, :meth:`lasting`)."""
        return cls(definition.id, duration_s, say=definition.say, done=definition.done, **code)

    def lasting(self, duration_s: float) -> WaitStep:
        """A copy that waits ``duration_s`` (a duration known only at run time)."""
        return replace(self, duration_s=duration_s)

    def format(self, **values: object) -> WaitStep:
        """A copy with ``{name}`` placeholders filled in ``say``; ``done`` gets them when it is spoken."""
        say = self.say.format(**values) if self.say else self.say
        return replace(self, say=say, values={**self.values, **values})

    def closing(self, result: TimerState) -> str | None:
        """``done``, filled with the build-time values."""
        return self.done.format(**self.values) if self.done else None

    async def run(self, io: StepIO) -> TimerState:
        return await timer(self.duration_s, interval_s=self.interval_s, on_state=io.report(self.id))
