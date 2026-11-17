# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A watch step: the camera confirms something, and the person helps when the camera alone cannot.

Two parts, sharing the step's context (:class:`~..judge.StepContext`):

- the fast check (:func:`~..verify.verify`): a frame every ``interval_s``, observed and labelled by fast models,
  done when ``required`` of the last ``window`` decidable frames are positive. No reasoning, never asks; it reads the
  context on every frame.
- the side escalation (:func:`~..escalate.gather`), only when the check has escalation guidance: when the check's
  latest reports are mostly unknown, a reasoning model reads them, infers what is missing, and asks the person (the
  answer goes into the context) or tells them what to do (bring the kettle into view). It never decides the step.

The escalation runs as a task of the step's own and ends with it; a failure in it is traced and does not fail the
step. ``Expect`` and :meth:`WatchStep.score` say how an evaluation case of a watch step is scored.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, ConfigDict
from xr_ai_app import clock, trace

from ..escalate import gather
from ..judge import Check, Judgment, StepContext
from ..step import Roles, StepIO, StepTrace, mentions
from ..verify import Verification, verify


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
    review_every_s: float = 20.0
    """How often the side escalation looks at the check's reports."""
    max_lines: int = 2
    """How many things the side escalation may say during one step."""


class Window(BaseModel):
    """A time window on the evaluation's scene clock."""

    model_config = ConfigDict(extra="forbid")
    after: float | None = None
    before: float | None = None


class WatchExpect(BaseModel):
    """What an evaluation case may expect of a watch step (each key optional)."""

    model_config = ConfigDict(extra="forbid")
    completes: bool | Window | None = None
    """``true``/``false``, or completes within a window of scene time."""
    asks: list[str] | None = None
    """It asks the person something that mentions one of these words (``[]``: any question)."""
    says: list[str] | None = None
    """It tells the person something (not a question) that mentions one of these words (``[]``: anything)."""
    quiet: bool = False
    """It neither asks nor tells the person anything."""


@dataclass(frozen=True, slots=True)
class WatchStep:
    """A camera check with its side escalation; see the module docstring."""

    id: str
    check: Check
    roles: Roles
    say: str | None = None
    known: BaseModel | None = None
    """Typed facts the step starts with (an instance of the check's ``needs``, from the app's settings)."""
    settings: WatchSettings = field(default_factory=WatchSettings)

    Expect = WatchExpect

    def format(self, **values: object) -> WatchStep:
        """A copy with ``{name}`` placeholders filled in the criteria and the spoken line (run-time targets)."""
        say = self.say.format(**values) if self.say else self.say
        return replace(self, check=self.check.format(**values), say=say)

    def knowing(self, known: BaseModel | None) -> WatchStep:
        """A copy that starts with ``known`` as its typed facts."""
        return replace(self, known=known)

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

        async def situation() -> str:
            # The same text live and in an evaluation: the step's own status, then what the app adds.
            lines = [f"The step '{self.id}' has been running for {clock.now() - started:.0f} s."]
            if around := await io.situation():
                lines.append(around)
            return "\n".join(lines)

        async def escalate() -> None:
            try:
                await gather(
                    context,
                    self.check,
                    reports,
                    role=self.roles.reason,
                    ask=io.ask,
                    say=io.say,
                    reader=self.roles.reason,
                    situation=situation,
                    every_s=settings.review_every_s,
                    max_questions=settings.max_lines,
                )
            except Exception as exc:  # noqa: BLE001 - the escalation never fails the step
                trace.emit("sop", self.id, op="escalation_failed", error=repr(exc))

        side = asyncio.create_task(escalate()) if self.check.escalation else None
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

    @staticmethod
    def score(trace: StepTrace, expect: WatchExpect) -> list[str]:
        """What is wrong with ``trace`` against ``expect`` (empty: the case passed)."""
        problems: list[str] = []
        done, at = trace.completed, trace.completed_at_s
        if isinstance(expect.completes, bool) and done != expect.completes:
            problems.append(f"completed at {at:.0f} s" if done and at is not None else "did not complete")
        elif isinstance(expect.completes, Window):
            w = expect.completes
            if not done or at is None:
                problems.append("did not complete")
            elif (w.after is not None and at < w.after) or (w.before is not None and at > w.before):
                problems.append(f"completed at {at:.0f} s, expected within [{w.after}, {w.before}]")
        asked, said = trace.asked(), trace.said()
        if expect.asks is not None and not any(mentions(line.text, expect.asks) for line in asked):
            got = "; ".join(repr(line.text) for line in asked) or "nothing"
            problems.append(f"asked {got}, expected a question about {expect.asks or 'anything'}")
        if expect.says is not None and not any(mentions(line.text, expect.says) for line in said):
            got = "; ".join(repr(line.text) for line in said) or "nothing"
            problems.append(f"said {got}, expected an instruction about {expect.says or 'anything'}")
        if expect.quiet and (asked or said):
            problems.append("said " + "; ".join(repr(line.text) for line in asked + said) + ", expected nothing")
        return problems
