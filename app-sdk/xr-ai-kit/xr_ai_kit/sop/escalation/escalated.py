# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A step with an escalator beside it: ``Escalated(step, escalator)``, itself a step.

Each run makes a new :class:`~.link.Link`, runs a copy of the step linked to it (``step.linked(link)``) and the
escalator beside it as a task of its own, and ends the escalator when the step ends. A failure in the escalator is
traced and never fails the step. The runner and the evaluation run it like any step, so an evaluation runs the
pairing that ships.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

from xr_ai_app import trace

from ..steps.step_api import StepIO, closing_line
from .link import Escalatable, EscalationIO, Escalator, Link

SCORING = f"{__package__}.escalation_scoring"
"""What an evaluation case may expect of the escalator (``asks``, ``says``, ``quiet``)."""


@dataclass(frozen=True, slots=True)
class Escalated:
    """``step`` (a step type that supports escalation, :class:`~.link.Escalatable`) with ``escalator`` (any
    :class:`~.link.Escalator`) beside it."""

    step: Any
    escalator: Escalator

    def __post_init__(self) -> None:
        if not isinstance(self.step, Escalatable):
            raise TypeError(f"{type(self.step).__name__} does not support escalation (it has no linked(link))")
        if not isinstance(self.escalator, Escalator):
            raise TypeError(f"{type(self.escalator).__name__} is not an escalator (it has no run(link, io))")

    @property
    def id(self) -> str:
        return self.step.id

    @property
    def say(self) -> str | None:
        return self.step.say

    @property
    def scoring(self) -> tuple[str, ...]:
        """The step's own scoring, and the escalator's."""
        own = getattr(self.step, "scoring", ())
        return (*((own,) if isinstance(own, str) else own), SCORING)

    def closing(self, result: Any) -> str | None:
        return closing_line(self.step, result)

    async def run(self, io: StepIO) -> Any:
        """Run the step and, beside it, the escalator, which talks to the person through ``io`` (it must also be an
        :class:`~.link.EscalationIO`, as the runner's and the evaluation's are)."""
        if not isinstance(io, EscalationIO):
            raise TypeError(f"an escalator talks to the person; {type(io).__name__} is not an EscalationIO")
        link = Link()
        side = asyncio.create_task(self._escalate(link, io))
        try:
            return await self.step.linked(link).run(io)
        finally:
            side.cancel()
            with contextlib.suppress(BaseException):
                await side

    async def _escalate(self, link: Link, io: EscalationIO) -> None:
        try:
            await self.escalator.run(link, io)
        except Exception as exc:  # noqa: BLE001 - the escalation never fails the step
            trace.emit("sop", self.id, op="escalation_failed", error=repr(exc))
