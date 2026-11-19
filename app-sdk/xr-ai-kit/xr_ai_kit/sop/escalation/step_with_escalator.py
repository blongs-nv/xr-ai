# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A step with an escalator beside it: ``StepWithEscalator(step, escalator)``, itself a step.

Each run makes a new :class:`~.escalation_channel.EscalationChannel`, runs the step with its end connected
(``step.run_with_escalation(environment, channel, inputs)``) and the escalator beside it as a task of its own, and ends
the escalator when the step ends. A failure in the escalator is traced and never fails the step. The runner and the
evaluation run it like any step, so an evaluation runs the pairing that ships.
"""

from __future__ import annotations

import asyncio
from typing import Any

from xr_ai_app import trace

from ..steps.step_api import SPEAKER, StepEnvironment
from .escalation_channel import EscalationChannel, Escalator, SupportsEscalation


class StepWithEscalator:
    """``step`` (a step type that supports escalation, :class:`~.escalation_channel.SupportsEscalation`) with
    ``escalator`` (any :class:`~.escalation_channel.Escalator`) beside it."""

    def __init__(self, step: SupportsEscalation, escalator: Escalator) -> None:
        if not isinstance(step, SupportsEscalation):
            raise TypeError(f"{type(step).__name__} does not support escalation (it has no run_with_escalation)")
        if not isinstance(escalator, Escalator):
            raise TypeError(f"{type(escalator).__name__} is not an escalator (it has no run(channel, environment))")
        self.step = step
        self.escalator = escalator
        self.id: str = step.id
        self.words = step.words
        """The step's words (what it does, its lines), as the runner's ``plan`` reads them."""
        self.inputs_schema = step.inputs_schema

    @property
    def scoring(self) -> tuple[str, ...]:
        """The step's own scoring, and the escalator's, if it has one."""
        return (*getattr(self.step, "scoring", ()), *getattr(self.escalator, "scoring", ()))

    @property
    def skippable(self) -> bool:
        return getattr(self.step, "skippable", True)

    def start_line(self, inputs: Any = None) -> str | None:
        return self.step.start_line(inputs)

    def done_line(self, inputs: Any, result: Any) -> str | None:
        return self.step.done_line(inputs, result)

    async def run(self, environment: StepEnvironment, inputs: Any = None) -> Any:
        """Run the step and, beside it, the escalator, which reaches the person through the same ``environment``."""
        channel = EscalationChannel()
        escalation = asyncio.create_task(self._escalate(channel, environment))
        try:
            return await self.step.run_with_escalation(environment, channel, inputs)
        finally:
            escalation.cancel()
            await asyncio.wait({escalation})  # let it end; its failures are traced in _escalate, never raised here

    async def _escalate(self, channel: EscalationChannel, environment: StepEnvironment) -> None:
        # create_task copied the caller's context, so this is set only in the escalator's own task: Run.ask and
        # Run.say read it to tag the escalator's lines, while the step's task still reads "step".
        SPEAKER.set("escalator")
        try:
            await self.escalator.run(channel, environment)
        except Exception as exc:  # noqa: BLE001 - the escalation never fails the step
            trace.emit("sop", self.id, op="escalation_failed", error=repr(exc))
