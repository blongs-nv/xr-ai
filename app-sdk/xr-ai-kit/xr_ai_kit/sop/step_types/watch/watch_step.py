# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A watch step: the camera confirms something.

The check (:func:`~.verify.verify`): a frame every ``interval_s``, described and labelled by fast models
(:class:`~.judge.WatchModels`), done when ``required`` of the last ``window`` decidable frames are positive. No
reasoning, never asks. It reads what it knows beyond the camera (:class:`~.judge.StepKnowledge`) on every frame.

It supports escalation (``...escalation``): run with an :class:`~...escalation.EscalationChannel`, it shows an
escalator a snapshot of itself (its task, its recent judgments as evidence, each with the frame it judged, and what it
knows) and adds what the escalator learns to its knowledge, which the next frame's decision reads. It knows nothing
else about the escalator.

Its words (:class:`WatchWords`): ``question`` and ``criteria``, and what every step says (``does``, ``start_line``,
``done_line``). Each run's inputs fill the ``{placeholders}`` in all of them; with an observation schema,
``done_line`` also gets ``observation`` (the observation that completed the check, an instance of that schema).
How an evaluation case of it is scored is in :mod:`.watch_scoring`.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel
from xr_ai_app import clock

from ...escalation.escalation_channel import EscalationChannel, Evidence, StepSnapshot
from ...steps.step_api import StepEnvironment, check_inputs, check_placeholders, fill_placeholders
from ...steps.step_words import StepWords
from .judge import Check, Criteria, Judgment, Knowledge, StepKnowledge, StructuredObservation, WatchModels
from .verify import DESCRIPTION_CHARS, Verification, WatchSettings, verify


class WatchWords(StepWords):
    """A watch step's words: what every step says, and its ``question`` and ``criteria``."""

    question: str
    """What the perceiving model describes in each frame."""
    criteria: Criteria
    """How the deciding model labels a description: ``question``, ``positive``, ``negative``, ``unknown``."""


class WatchStep:
    """A camera check; see the module docstring.

    Built from the step's id, its ``words`` and the ``models`` it uses. ``inputs_schema`` is what each run starts with
    (the values its words name, such as a target temperature; ``None``: nothing). ``facts_schema`` is the typed facts
    it may need that the camera may not show (an escalator beside it may ask the person for them); give every field a
    default, since the person may state only some. ``observation`` reads each frame into a schema
    (:class:`~.judge.StructuredObservation`).
    """

    scoring: ClassVar[tuple[str, ...]] = (f"{__package__}.watch_scoring",)

    def __init__(
        self,
        step_id: str,
        words: WatchWords,
        models: WatchModels,
        *,
        inputs_schema: type[BaseModel] | None = None,
        facts_schema: type[BaseModel] | None = None,
        observation: StructuredObservation | None = None,
        settings: WatchSettings | None = None,
    ) -> None:
        self.id = step_id
        self.words = words
        self.inputs_schema = inputs_schema
        self.facts_schema = facts_schema
        self.observation = observation
        self.models = models
        self.settings = settings or WatchSettings()
        criteria = words.criteria
        for where, template in (
            ("question", words.question),
            ("criteria.question", criteria.question),
            ("criteria.positive", criteria.positive),
            ("criteria.negative", criteria.negative),
            ("criteria.unknown", criteria.unknown),
            ("start_line", words.start_line),
        ):
            check_placeholders(self.id, where, template, inputs_schema)
        done_line_also = ("observation",) if observation is not None else ()
        check_placeholders(self.id, "done_line", words.done_line, inputs_schema, also=done_line_also)

    def start_line(self, inputs: BaseModel | None = None) -> str | None:
        return fill_placeholders(self.words.start_line, inputs)

    def done_line(self, inputs: BaseModel | None, result: Verification) -> str | None:
        """``done_line``, filled with the inputs and the observation that completed the check."""
        observation = next((j.parsed for j in reversed(result.judgments) if j.label == "positive"), None)
        return fill_placeholders(self.words.done_line, inputs, observation=observation)

    async def run(self, environment: StepEnvironment, inputs: BaseModel | None = None) -> Verification:
        """Watch until the check completes; returns its final state. Cancelling the task ends the step."""
        return await self._watch(environment, inputs, None)

    async def run_with_escalation(
        self, environment: StepEnvironment, channel: EscalationChannel, inputs: BaseModel | None = None
    ) -> Verification:
        """:meth:`run`, showing ``channel`` a snapshot of the step and taking what the escalator adds."""
        return await self._watch(environment, inputs, channel)

    async def _watch(
        self, environment: StepEnvironment, inputs: BaseModel | None, channel: EscalationChannel | None
    ) -> Verification:
        check_inputs(self.id, self.inputs_schema, inputs)
        criteria = self.words.criteria
        check = Check(
            fill_placeholders(self.words.question, inputs) or "",
            Criteria(
                fill_placeholders(criteria.question, inputs) or "",
                fill_placeholders(criteria.positive, inputs) or "",
                fill_placeholders(criteria.negative, inputs) or "",
                fill_placeholders(criteria.unknown, inputs) or "",
            ),
            facts_schema=self.facts_schema,
            observation=self.observation,
            inputs=inputs,
        )
        knowledge = StepKnowledge()
        latest: Verification | None = None

        async def on_state(state: Verification) -> None:
            nonlocal latest
            latest = state
            await environment.report_progress(self.id, state)

        if channel is not None:
            started = clock.now()
            channel.connect_step(
                snapshot=lambda: watch_step_snapshot(
                    self.id, check, latest.judgments if latest else (), knowledge.knowledge, started
                ),
                add_context=lambda added: knowledge.add(added.note, added.facts),
            )
        return await verify(
            environment.frames,  # type: ignore[arg-type]  # any Frames: the participant's, or a recorded scene
            check,
            self.models,
            self.settings,
            knowledge=knowledge,
            on_state=on_state,
        )


def watch_step_snapshot(
    step_id: str,
    check: Check,
    judgments: tuple[Judgment, ...] | list[Judgment],
    knowledge: Knowledge,
    started_s: float,
) -> StepSnapshot:
    """A watch step as its escalator sees it: its check as its task, its judgments as evidence, its knowledge as what
    it knows."""
    criteria = check.criteria
    return StepSnapshot(
        step_id,
        started_s,
        task=(
            "A fast camera check is verifying one step of a guided procedure, from the person's first-person view.",
            f"What it verifies: {criteria.question}",
            f"It completes when: {criteria.positive}",
            f"It cannot decide when: {criteria.unknown}",
        ),
        evidence=tuple(
            Evidence(
                j.label,
                j.description[:DESCRIPTION_CHARS] + (f" ({j.reason})" if j.reason else ""),
                j.label != "unknown",
                j.frame,
            )
            for j in judgments
        ),
        knowledge_text=knowledge.text(),
        facts_schema=check.facts_schema,
    )
