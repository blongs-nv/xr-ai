# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A watch step: the camera confirms something.

The check (:func:`~.verify.verify`): a frame every ``interval_s``, described and labelled by fast models
(:class:`~.judge.WatchModels`), done when ``required`` of the last ``window`` decidable frames are positive. No
reasoning, never asks. It reads its context (:class:`~.judge.StepContext`: the typed facts it started with, and
anything added since) on every frame.

It supports escalation (``...escalation``): run with an :class:`~...escalation.EscalationChannel`, it shows an
escalator a snapshot of itself (its task, its recent judgments as evidence, each with the frame it judged, and what it
knows) and adds the context the escalator sends to its own, which the next frame's decision reads. It knows nothing
else about the escalator.

Its definition in the procedure's YAML: the ``watch`` section (:class:`WatchSection`: ``question`` and ``criteria``),
``start_line``, and ``done_line``. ``values`` fill the ``{placeholders}`` in all of them; ``done_line`` also gets
``observation`` (the observation that completed the check, an instance of its observation schema when it has one).
How an evaluation case of it is scored is in :mod:`.watch_scoring`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from pydantic import BaseModel, ConfigDict
from xr_ai_app import clock

from ...escalation.escalation_channel import EscalationChannel, Evidence, StepSnapshot
from ...steps.step_api import StepEnvironment, fill_placeholders
from ...steps.step_definitions import StepDefinition
from .judge import Check, Criteria, Facts, Judgment, StepContext, StructuredObservation, WatchModels
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


class WatchSection(BaseModel):
    """A watch step's section of its definition (``watch:``)."""

    model_config = ConfigDict(extra="forbid")
    question: str
    """What the perceiving model describes in each frame."""
    criteria: Criteria
    """How the deciding model labels a description: ``instructions``, ``positive``, ``negative``, ``unknown``."""


class WatchStep:
    """A camera check; see the module docstring.

    Built from the step's ``definition`` and the ``models`` it uses. ``values`` fill the placeholders in its words
    (targets known only at run time). ``facts_schema`` is the typed facts it may need that the camera may not show,
    and ``known_facts`` (an instance of it) what is known from the start, such as an app setting. ``observation``
    reads each frame into a schema (:class:`~.judge.StructuredObservation`).
    """

    scoring: ClassVar[str] = f"{__package__}.watch_scoring"
    section: ClassVar[str] = "watch"

    def __init__(
        self,
        definition: StepDefinition,
        models: WatchModels,
        *,
        values: Mapping[str, object] | None = None,
        facts_schema: type[BaseModel] | None = None,
        known_facts: BaseModel | None = None,
        observation: StructuredObservation | None = None,
        settings: WatchSettings | None = None,
    ) -> None:
        if known_facts is not None and (facts_schema is None or not isinstance(known_facts, facts_schema)):
            raise TypeError(f"step {definition.id!r}: known_facts must be an instance of its facts_schema")
        self.values = dict(values or {})
        words = definition.section(self.section, WatchSection)
        criteria = words.criteria
        self.id = definition.id
        self.start_line = fill_placeholders(definition.start_line, self.values)
        self.check = Check(
            words.question.format(**self.values),
            Criteria(
                criteria.instructions.format(**self.values),
                criteria.positive.format(**self.values),
                criteria.negative.format(**self.values),
                criteria.unknown.format(**self.values),
            ),
            facts_schema=facts_schema,
            observation=observation,
        )
        self.models = models
        self.known_facts = known_facts
        self.settings = settings or WatchSettings()
        self._done_line = definition.done_line

    def done_line(self, result: Verification) -> str | None:
        """``done_line``, filled with the values and the observation that completed the check."""
        observation = next((j.parsed for j in reversed(result.judgments) if j.label == "positive"), None)
        return fill_placeholders(self._done_line, {**self.values, "observation": observation})

    async def run(self, environment: StepEnvironment) -> Verification:
        """Watch until the check completes; returns its final state. Cancelling the task ends the step."""
        return await self._watch(environment, None)

    async def run_with_escalation(self, environment: StepEnvironment, channel: EscalationChannel) -> Verification:
        """:meth:`run`, showing ``channel`` a snapshot of the step and taking the context it adds."""
        return await self._watch(environment, channel)

    async def _watch(self, environment: StepEnvironment, channel: EscalationChannel | None) -> Verification:
        settings = self.settings
        context = StepContext(self.known_facts)
        latest: list[Verification] = []

        async def on_state(state: Verification) -> None:
            latest[:] = [state]
            await environment.report_progress(self.id, state)

        if channel is not None:
            started = clock.now()
            channel.connect_step(
                snapshot=lambda: watch_step_snapshot(
                    self.id, self.check, latest[0].judgments if latest else (), context.facts, started
                ),
                add_context=lambda added: context.add(added.note, added.facts),
            )
        return await verify(
            environment.frames,  # type: ignore[arg-type]  # any Frames: the participant's, or a recorded scene
            self.check,
            self.models,
            context=context,
            required=settings.required,
            window=settings.window,
            interval_s=settings.interval_s,
            timeout_s=settings.timeout_s,
            on_state=on_state,
        )


def watch_step_snapshot(
    step_id: str, check: Check, judgments: tuple[Judgment, ...] | list[Judgment], facts: Facts, started_s: float
) -> StepSnapshot:
    """A watch step as its escalator sees it: its check as its task, its judgments as evidence, its facts as what it
    knows."""
    criteria = check.criteria
    return StepSnapshot(
        step_id,
        started_s,
        task=(
            "A fast camera check is verifying one step of a guided procedure, from the person's first-person view.",
            f"What it verifies: {criteria.instructions}",
            f"It completes when: {criteria.positive}",
            f"It cannot decide when: {criteria.unknown}",
        ),
        evidence=tuple(
            Evidence(
                j.label, j.description[:300] + (f" ({j.reason})" if j.reason else ""), j.label != "unknown", j.frame
            )
            for j in judgments
        ),
        known=facts.text(),
        facts_schema=check.facts_schema,
    )
