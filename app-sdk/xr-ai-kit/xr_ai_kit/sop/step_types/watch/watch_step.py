# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A watch step: the camera confirms something.

The check (:func:`~.verify.verify`): a frame every ``interval_s``, observed and labelled by fast models, done when
``required`` of the last ``window`` decidable frames are positive. No reasoning, never asks. It reads its context
(:class:`~.judge.StepContext`: the typed facts it started with, and anything added since) on every frame.

It supports escalation (``..escalation``): linked to a run's :class:`~...escalation.Link`, it shows an escalator its
task, its recent judgments, each with the frame it judged, and what it knows (outbound), and adds what the escalator
sends to its context (inbound), which the next frame's decision reads. It knows nothing else about the escalator.

Its definition in the procedure's YAML: the ``watch`` section (:class:`WatchWords`: ``question`` and ``criteria``),
and ``done``, filled with the build-time values and ``observation`` (the observation that completed it, an instance
of the check's ``observation_model`` when it has one). How an evaluation case of it is scored is in
:mod:`.watch_scoring`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from xr_ai_app import Role, clock

from ...escalation.link import Link, Outbound, Report
from ...steps.step_api import StepIO
from ...steps.step_definitions import StepDefinition
from .judge import Check, Criteria, Facts, Judgment, StepContext
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


class WatchWords(BaseModel):
    """A watch step's section of its definition (``watch:``)."""

    model_config = ConfigDict(extra="forbid")
    question: str
    """What the observing model describes in each frame."""
    criteria: Criteria
    """How the deciding model labels a description: ``instructions``, ``positive``, ``negative``, ``unknown``."""


@dataclass(frozen=True, slots=True)
class WatchStep:
    """A camera check; see the module docstring.

    ``observe`` describes each frame (a vision model); ``decide`` labels the description (a fast classifier).
    """

    scoring: ClassVar[str] = f"{__package__}.watch_scoring"
    section: ClassVar[str] = "watch"

    id: str
    check: Check
    observe: Role
    decide: Role
    say: str | None = None
    done: str | None = None
    """Spoken when it completes (see the module docstring for its placeholders)."""
    values: Mapping[str, object] = field(default_factory=dict)
    """The build-time values given to :meth:`format`, for ``done``."""
    known: BaseModel | None = None
    """Typed facts the step starts with (an instance of the check's ``needs``, from the app's settings)."""
    settings: WatchSettings = field(default_factory=WatchSettings)
    link: Link | None = None
    """The run's link to an escalator (set by :meth:`linked`, per run; ``None``: no escalator)."""

    @classmethod
    def define(
        cls,
        definition: StepDefinition,
        *,
        observe: Role,
        decide: Role,
        settings: WatchSettings | None = None,
        **check: Any,
    ) -> WatchStep:
        """The step from its definition. ``check`` is what code adds to the :class:`~.judge.Check` (``needs``,
        ``observation_model``, ``can_assess``, ``reconcile``)."""
        words = definition.section(cls.section, WatchWords)
        return cls(
            definition.id,
            Check(words.question, words.criteria, **check),
            observe,
            decide,
            say=definition.say,
            done=definition.done,
            settings=settings or WatchSettings(),
        )

    def format(self, **values: object) -> WatchStep:
        """A copy with ``{name}`` placeholders filled in the criteria and ``say`` (run-time targets); ``done`` gets
        them when it is spoken."""
        say = self.say.format(**values) if self.say else self.say
        return replace(self, check=self.check.format(**values), say=say, values={**self.values, **values})

    def knowing(self, known: BaseModel | None) -> WatchStep:
        """A copy that starts with ``known`` as its typed facts."""
        return replace(self, known=known)

    def linked(self, link: Link) -> WatchStep:
        """A copy that shows ``link`` its outbound context and takes inbound context from it (escalation)."""
        return replace(self, link=link)

    def closing(self, result: Verification) -> str | None:
        """``done``, filled with the build-time values and the observation that completed the check."""
        if not self.done:
            return None
        observation = next((j.parsed for j in reversed(result.judgments) if j.label == "positive"), None)
        return self.done.format(**{**self.values, "observation": observation})

    async def run(self, io: StepIO) -> Verification:
        """Watch until the check completes; returns its final state. Cancelling the task ends the step."""
        settings = self.settings
        context = StepContext(self.known)
        latest: list[Verification] = []
        report = io.report(self.id)

        async def on_state(state: Verification) -> None:
            latest[:] = [state]
            await report(state)

        if self.link is not None:
            started = clock.now()
            self.link.attach(
                lambda: outbound(self.id, self.check, latest[0].judgments if latest else (), context.facts, started),
                lambda inbound: context.add(inbound.note, inbound.known),
            )
        return await verify(
            io.frames,  # type: ignore[arg-type]  # any Frames: the participant's, or a recorded scene
            self.check,
            observe=self.observe,
            decide=self.decide,
            context=context,
            required=settings.required,
            window=settings.window,
            interval_s=settings.interval_s,
            timeout_s=settings.timeout_s,
            on_state=on_state,
        )


def outbound(
    step_id: str, check: Check, judgments: tuple[Judgment, ...] | list[Judgment], facts: Facts, started_s: float
) -> Outbound:
    """A watch step's outbound context: its check as its task, its judgments as reports, its facts as what it knows."""
    criteria = check.criteria
    return Outbound(
        step_id,
        started_s,
        task=(
            "A fast camera check is verifying one step of a guided procedure, from the person's first-person view.",
            f"What it verifies: {criteria.instructions}",
            f"It completes when: {criteria.positive}",
            f"It cannot decide when: {criteria.unknown}",
        ),
        reports=tuple(
            Report(j.label, j.description[:300] + (f" ({j.reason})" if j.reason else ""), j.label != "unknown", j.frame)
            for j in judgments
        ),
        known=facts.text(),
        needs=check.needs,
    )
