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
``start_line``, and ``done_line``. Each run's inputs fill the ``{placeholders}`` in all of them; ``done_line`` also
gets ``observation`` (the observation that completed the check, an instance of its observation schema when it has
one).
How an evaluation case of it is scored is in :mod:`.watch_scoring`.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict
from xr_ai_app import clock

from ...escalation.escalation_channel import EscalationChannel, Evidence, StepSnapshot
from ...steps.step_api import StepEnvironment, check_inputs, check_placeholders, fill_placeholders
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

    Built from the step's ``definition`` and the ``models`` it uses. ``inputs_schema`` is what each run starts with
    (the values its words name, such as a target temperature; ``None``: nothing). ``facts_schema`` is the typed facts
    it may need that the camera may not show (an escalator beside it may ask the person for them). ``observation``
    reads each frame into a schema (:class:`~.judge.StructuredObservation`).
    """

    scoring: ClassVar[str] = f"{__package__}.watch_scoring"
    section: ClassVar[str] = "watch"

    def __init__(
        self,
        definition: StepDefinition,
        models: WatchModels,
        *,
        inputs_schema: type[BaseModel] | None = None,
        facts_schema: type[BaseModel] | None = None,
        observation: StructuredObservation | None = None,
        settings: WatchSettings | None = None,
    ) -> None:
        words = definition.section(self.section, WatchSection)
        self.id = definition.id
        self.inputs_schema = inputs_schema
        self.check = Check(words.question, words.criteria, facts_schema=facts_schema, observation=observation)
        """The check as written; each run fills its placeholders from that run's inputs."""
        self.models = models
        self.settings = settings or WatchSettings()
        self._start_line = definition.start_line
        self._done_line = definition.done_line
        criteria = words.criteria
        for where, template in (
            ("question", words.question),
            ("criteria", criteria.instructions),
            ("criteria", criteria.positive),
            ("criteria", criteria.negative),
            ("criteria", criteria.unknown),
            ("start_line", definition.start_line),
        ):
            check_placeholders(self.id, where, template, inputs_schema)
        check_placeholders(self.id, "done_line", definition.done_line, inputs_schema, also=("observation",))

    def start_line(self, inputs: BaseModel | None = None) -> str | None:
        return fill_placeholders(self._start_line, inputs)

    def done_line(self, inputs: BaseModel | None, result: Verification) -> str | None:
        """``done_line``, filled with the inputs and the observation that completed the check."""
        observation = next((j.parsed for j in reversed(result.judgments) if j.label == "positive"), None)
        return fill_placeholders(self._done_line, inputs, observation=observation)

    def check_for(self, inputs: BaseModel | None) -> Check:
        """The check with its placeholders filled from ``inputs``."""
        criteria = self.check.criteria
        return Check(
            fill_placeholders(self.check.question, inputs) or "",
            Criteria(
                fill_placeholders(criteria.instructions, inputs) or "",
                fill_placeholders(criteria.positive, inputs) or "",
                fill_placeholders(criteria.negative, inputs) or "",
                fill_placeholders(criteria.unknown, inputs) or "",
            ),
            facts_schema=self.check.facts_schema,
            observation=self.check.observation,
        )

    async def run(self, environment: StepEnvironment, inputs: BaseModel | None = None) -> Verification:
        """Watch until the check completes; returns its final state. Cancelling the task ends the step."""
        return await self._watch(environment, inputs, None)

    async def run_with_escalation(
        self, environment: StepEnvironment, channel: EscalationChannel, inputs: BaseModel | None = None
    ) -> Verification:
        """:meth:`run`, showing ``channel`` a snapshot of the step and taking the context it adds."""
        return await self._watch(environment, inputs, channel)

    async def _watch(
        self, environment: StepEnvironment, inputs: BaseModel | None, channel: EscalationChannel | None
    ) -> Verification:
        check_inputs(self.id, self.inputs_schema, inputs)
        settings = self.settings
        check = self.check_for(inputs)
        context = StepContext()
        latest: list[Verification] = []

        async def on_state(state: Verification) -> None:
            latest[:] = [state]
            await environment.report_progress(self.id, state)

        if channel is not None:
            started = clock.now()
            channel.connect_step(
                snapshot=lambda: watch_step_snapshot(
                    self.id, check, latest[0].judgments if latest else (), context.facts, started
                ),
                add_context=lambda added: context.add(added.note, added.facts),
            )
        return await verify(
            environment.frames,  # type: ignore[arg-type]  # any Frames: the participant's, or a recorded scene
            check,
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
