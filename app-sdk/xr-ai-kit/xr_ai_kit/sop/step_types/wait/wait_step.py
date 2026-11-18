# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A wait step: a countdown on the core clock (:func:`~.timer.timer`); no camera, no models.

Its duration is either fixed (``WaitStep(definition, duration_s=90)``) or each run's input (an ``inputs_schema``
that extends :class:`WaitInputs`, known only once earlier steps have run). Its definition has no section of its own:
``start_line`` and ``done_line``, whose placeholders name fields or properties of its inputs."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from ...steps.step_api import StepEnvironment, check_inputs, check_placeholders, fill_placeholders
from ...steps.step_definitions import StepDefinition
from .timer import TimerState, timer


class WaitInputs(BaseModel):
    """What a wait step with a run-time duration starts with; an app extends it with what its words name."""

    duration_s: float = Field(gt=0)


class WaitStep:
    """Wait ``duration_s`` seconds (fixed, or each run's input), reporting the time left every ``interval_s``."""

    scoring: ClassVar[str] = f"{__package__}.wait_scoring"

    def __init__(
        self,
        definition: StepDefinition,
        *,
        duration_s: float | None = None,
        inputs_schema: type[WaitInputs] | None = None,
        interval_s: float = 1.0,
    ) -> None:
        if (duration_s is None) == (inputs_schema is None):
            raise TypeError(f"step {definition.id!r}: give a fixed duration_s or an inputs_schema (a WaitInputs)")
        if inputs_schema is not None and not issubclass(inputs_schema, WaitInputs):
            raise TypeError(f"step {definition.id!r}: its inputs_schema must extend WaitInputs")
        self.id = definition.id
        self.inputs_schema = inputs_schema
        self.duration_s = duration_s
        self.interval_s = interval_s
        self._start_line = definition.start_line
        self._done_line = definition.done_line
        check_placeholders(self.id, "start_line", definition.start_line, inputs_schema)
        check_placeholders(self.id, "done_line", definition.done_line, inputs_schema)

    def start_line(self, inputs: WaitInputs | None = None) -> str | None:
        return fill_placeholders(self._start_line, inputs)

    def done_line(self, inputs: WaitInputs | None, result: TimerState) -> str | None:
        return fill_placeholders(self._done_line, inputs)

    async def run(self, environment: StepEnvironment, inputs: WaitInputs | None = None) -> TimerState:
        check_inputs(self.id, self.inputs_schema, inputs)
        duration_s = inputs.duration_s if inputs is not None else self.duration_s
        assert duration_s is not None  # noqa: S101 - the constructor requires one of the two

        async def on_state(state: TimerState) -> None:
            await environment.report_progress(self.id, state)

        return await timer(duration_s, interval_s=self.interval_s, on_state=on_state)
