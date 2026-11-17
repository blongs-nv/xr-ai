# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A watch step: the camera confirms something, and the person helps when the camera alone cannot.

Two parts, sharing the step's context (:class:`~.judge.StepContext`):

- the fast check (:func:`~.verify.verify`): a frame every ``interval_s``, observed and labelled by fast models,
  done when ``required`` of the last ``window`` decidable frames are positive. No reasoning, never asks; it reads the
  context on every frame.
- the step's escalator (:class:`~.escalator.Escalator`), only when one is attached (:meth:`WatchStep.escalating`):
  a separate agent, configured per step, that reads the check's reports and, when they are mostly unknown, asks the
  person (the answer goes into the context) or tells them what to do (bring the kettle into view). It never decides
  the step.

The escalator runs as a task of the step's own and ends with it; a failure in it is traced and does not fail the
step. How an evaluation case of a watch step is scored is in :mod:`.watch_scoring`.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, ConfigDict
from xr_ai_app import clock, trace

from ..step_api import Roles, StepIO
from .escalator import Escalator, Watched
from .judge import Check, Judgment, StepContext
from .verify import Verification, verify


class WatchSettings(BaseModel):
    """How a watch step watches (an app's settings may embed this)."""

    model_config = ConfigDict(extra="forbid")
    required: int = 2
    """Positives needed ..."""
    window: int = 4
    """... among the last this many decidable frames."""
    interval_s: float = 2.0
    """How often a frame is judged."""
    timeout_s: float = 30.0
    """Bound on one model call."""


@dataclass(frozen=True, slots=True)
class WatchStep:
    """A camera check, with an optional escalator; see the module docstring."""

    id: str
    check: Check
    roles: Roles
    say: str | None = None
    known: BaseModel | None = None
    """Typed facts the step starts with (an instance of the check's ``needs``, from the app's settings)."""
    settings: WatchSettings = field(default_factory=WatchSettings)
    escalator: Escalator | None = None
    """The step's side escalation (``None``: the step never reasons, never asks)."""

    def format(self, **values: object) -> WatchStep:
        """A copy with ``{name}`` placeholders filled in the criteria and the spoken line (run-time targets)."""
        say = self.say.format(**values) if self.say else self.say
        return replace(self, check=self.check.format(**values), say=say)

    def knowing(self, known: BaseModel | None) -> WatchStep:
        """A copy that starts with ``known`` as its typed facts."""
        return replace(self, known=known)

    def escalating(self, escalator: Escalator | None) -> WatchStep:
        """A copy with ``escalator`` as its side escalation (``None``: without one)."""
        return replace(self, escalator=escalator)

    async def run(self, io: StepIO) -> Verification:
        """Watch until the check completes; returns its final state. Cancelling the task ends the step."""
        settings = self.settings
        context = StepContext(self.known)
        latest: list[Verification] = []
        report = io.report(self.id)
        started = clock.now()

        async def on_state(state: Verification) -> None:
            latest[:] = [state]
            await report(state)

        def reports() -> tuple[Judgment, ...]:
            return latest[0].judgments if latest else ()

        async def escalate(escalator: Escalator) -> None:
            watched = Watched(self.id, self.check, context, reports, started)
            try:
                await escalator.run(watched, io)
            except Exception as exc:  # noqa: BLE001 - the escalation never fails the step
                trace.emit("sop", self.id, op="escalation_failed", error=repr(exc))

        side = asyncio.create_task(escalate(self.escalator)) if self.escalator is not None else None
        try:
            return await verify(
                io.frames,  # type: ignore[arg-type]  # any Frames: the participant's, or a recorded scene
                self.check,
                observe=self.roles.observe,
                decide=self.roles.decide,
                context=context,
                required=settings.required,
                window=settings.window,
                interval_s=settings.interval_s,
                timeout_s=settings.timeout_s,
                on_state=on_state,
            )
        finally:
            if side is not None:
                side.cancel()
                with contextlib.suppress(BaseException):
                    await side
