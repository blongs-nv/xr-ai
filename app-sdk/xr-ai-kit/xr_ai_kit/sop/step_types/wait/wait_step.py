# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A wait step: a countdown on the core clock (:func:`~.timer.timer`); no camera, no models.

Its definition has no section of its own: ``start_line`` and ``done_line``, filled with ``values``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ...steps.step_api import StepEnvironment, fill_placeholders
from ...steps.step_definitions import StepDefinition
from .timer import TimerState, timer


class WaitStep:
    """Wait ``duration_s`` seconds, reporting the time left every ``interval_s``."""

    scoring: ClassVar[str] = f"{__package__}.wait_scoring"

    def __init__(
        self,
        definition: StepDefinition,
        duration_s: float,
        *,
        values: Mapping[str, object] | None = None,
        interval_s: float = 1.0,
    ) -> None:
        self.values = dict(values or {})
        self.id = definition.id
        self.start_line = fill_placeholders(definition.start_line, self.values)
        self.duration_s = duration_s
        self.interval_s = interval_s
        self._done_line = definition.done_line

    def done_line(self, result: TimerState) -> str | None:
        """``done_line``, filled with the values."""
        return fill_placeholders(self._done_line, self.values)

    async def run(self, environment: StepEnvironment) -> TimerState:
        async def on_state(state: TimerState) -> None:
            await environment.report_progress(self.id, state)

        return await timer(self.duration_s, interval_s=self.interval_s, on_state=on_state)
