# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A step with an escalator beside it: ``StepWithEscalator(step, escalator)``, itself a step.

Each run makes a new :class:`~.escalation_channel.EscalationChannel`, runs the step with its end connected
(``step.run_with_escalation(environment, channel)``) and the escalator beside it as a task of its own, and ends the
escalator when the step ends. A failure in the escalator is traced and never fails the step. The runner and the
evaluation run it like any step, so an evaluation runs the pairing that ships.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from xr_ai_app import trace

from ..steps.step_api import StepEnvironment
from .escalation_channel import (
    ESCALATOR_SPEAKING,
    EscalationChannel,
    Escalator,
    EscalatorEnvironment,
    SupportsEscalation,
)

SCORING = f"{__package__}.escalation_scoring"
"""What an evaluation case may expect of the escalator (``asks``, ``says``, ``quiet``)."""


class StepWithEscalator:
    """``step`` (a step type that supports escalation, :class:`~.escalation_channel.SupportsEscalation`) with
    ``escalator`` (any :class:`~.escalation_channel.Escalator`) beside it."""

    def __init__(self, step: Any, escalator: Escalator) -> None:
        if not isinstance(step, SupportsEscalation):
            raise TypeError(f"{type(step).__name__} does not support escalation (it has no run_with_escalation)")
        if not isinstance(escalator, Escalator):
            raise TypeError(f"{type(escalator).__name__} is not an escalator (it has no run(channel, environment))")
        self.step = step
        self.escalator = escalator
        self.id: str = step.id
        self.start_line: str | None = step.start_line

    @property
    def scoring(self) -> tuple[str, ...]:
        """The step's own scoring, and the escalator's."""
        own = getattr(self.step, "scoring", ())
        return (*((own,) if isinstance(own, str) else own), SCORING)

    def done_line(self, result: Any) -> str | None:
        return self.step.done_line(result)

    async def run(self, environment: StepEnvironment) -> Any:
        """Run the step and, beside it, the escalator, which reaches the person through ``environment`` (it must also
        be an :class:`~.escalation_channel.EscalatorEnvironment`, as the runner's and the evaluation's are)."""
        if not isinstance(environment, EscalatorEnvironment):
            raise TypeError(f"an escalator needs an EscalatorEnvironment; {type(environment).__name__} is not one")
        channel = EscalationChannel()
        escalation = asyncio.create_task(self._escalate(channel, environment))
        try:
            return await self.step.run_with_escalation(environment, channel)
        finally:
            escalation.cancel()
            with contextlib.suppress(BaseException):
                await escalation

    async def _escalate(self, channel: EscalationChannel, environment: EscalatorEnvironment) -> None:
        ESCALATOR_SPEAKING.set(True)  # this task's own context: what it says is the escalator's
        try:
            await self.escalator.run(channel, environment)
        except Exception as exc:  # noqa: BLE001 - the escalation never fails the step
            trace.emit("sop", self.id, op="escalation_failed", error=repr(exc))
