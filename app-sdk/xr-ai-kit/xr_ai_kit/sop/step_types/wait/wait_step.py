# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A wait step: a countdown on the core clock (:func:`~.timer.timer`); no camera, no models.

Its duration is either fixed (``WaitStep("warm_up", words, duration_s=90)``) or each run's input (an
``inputs_schema`` that extends :class:`WaitInputs`, known only once earlier steps have run). Its words are what every
step says (:class:`~...steps.step_words.StepWords`): ``start_line`` and ``done_line``, whose placeholders name fields
or properties of its inputs, or, with a fixed duration, ``{duration_s}`` and ``{minutes}``."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from ...steps.step_api import StepEnvironment, check_inputs, check_placeholders, fill_placeholders
from ...steps.step_words import StepWords
from .timer import TimerState, timer


class WaitInputs(BaseModel):
    """What a wait step with a run-time duration starts with; an app extends it with what its words name."""

    duration_s: float = Field(gt=0)


class WaitStep:
    """Wait ``duration_s`` seconds (fixed, or each run's input), reporting the time left every ``interval_s`` (its one
    setting: a countdown has nothing else to tune)."""

    scoring: ClassVar[tuple[str, ...]] = (f"{__package__}.wait_scoring",)

    def __init__(
        self,
        step_id: str,
        words: StepWords,
        *,
        duration_s: float | None = None,
        inputs_schema: type[WaitInputs] | None = None,
        interval_s: float = 1.0,
    ) -> None:
        if (duration_s is None) == (inputs_schema is None):
            raise TypeError(f"step {step_id!r}: give a fixed duration_s or an inputs_schema (a WaitInputs)")
        if inputs_schema is not None and not issubclass(inputs_schema, WaitInputs):
            raise TypeError(f"step {step_id!r}: its inputs_schema must extend WaitInputs")
        self.id = step_id
        self.words = words
        self.inputs_schema = inputs_schema
        self.duration_s = duration_s
        self.interval_s = interval_s
        self._fixed = {} if duration_s is None else {"duration_s": f"{duration_s:g}", "minutes": f"{duration_s / 60:g}"}
        """What a fixed duration's words may name ({duration_s}, {minutes})."""
        check_placeholders(self.id, "start_line", words.start_line, inputs_schema, also=tuple(self._fixed))
        check_placeholders(self.id, "done_line", words.done_line, inputs_schema, also=tuple(self._fixed))

    def start_line(self, inputs: WaitInputs | None = None) -> str | None:
        return fill_placeholders(self.words.start_line, inputs, **self._fixed)

    def done_line(self, inputs: WaitInputs | None, result: TimerState) -> str | None:
        return fill_placeholders(self.words.done_line, inputs, **self._fixed)

    async def run(self, environment: StepEnvironment, inputs: WaitInputs | None = None) -> TimerState:
        check_inputs(self.id, self.inputs_schema, inputs)
        duration_s = inputs.duration_s if inputs is not None else self.duration_s
        assert duration_s is not None  # noqa: S101 - the constructor requires one of the two

        async def on_state(state: TimerState) -> None:
            await environment.report_progress(self.id, state)

        return await timer(duration_s, interval_s=self.interval_s, on_state=on_state)
