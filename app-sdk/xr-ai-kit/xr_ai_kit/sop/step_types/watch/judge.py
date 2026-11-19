# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One camera frame judged against a check: the unit every visual check is built from.

A :class:`Check` is what one run of a step verifies: the question asked of each frame, the criteria for done, and,
optionally, the typed facts it may need that the camera cannot show (``facts_schema``; a step's escalator may ask the
person for them) and how to read each frame into a schema (:class:`StructuredObservation`). The perceiving model
(:class:`WatchModels`) describes the frame (free text, or an instance of the schema); the deciding model reads that
description, with the :class:`Knowledge` the step has beyond the camera, and picks ``positive``, ``negative`` or
``unknown`` against the criteria. Keeping the two apart means the observation can be read back and evaluated on its
own, and the same function runs live (inside :func:`verify`) and offline (an evaluation over recorded frames).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel
from xr_ai_app import Frame, Role, StructuredOutputError

Label = Literal["positive", "negative", "unknown"]
LABELS: tuple[Label, ...] = ("positive", "negative", "unknown")


@dataclass(frozen=True, slots=True)
class Criteria:
    """What counts as done for one check: the question the deciding role answers and one criterion per label."""

    question: str
    """The question the deciding role answers about an observation (for example "Is there water in the kettle?").
    Not the step's ``question``, which the perceiving model answers about the frame."""

    positive: str
    """When the observation establishes the step is done."""

    negative: str
    """When the observation establishes the step is not done."""

    unknown: str
    """When the observation cannot tell (nothing relevant in view, obscured, a picture of the thing)."""

    def choices(self) -> dict[str, str]:
        """The criteria as ``role.decide`` choices."""
        return {"positive": self.positive, "negative": self.negative, "unknown": self.unknown}


@dataclass(frozen=True, slots=True)
class Judgment:
    """One frame's verdict, with the observation it was based on."""

    label: Label
    description: str
    """What the observing role said about the frame (JSON text when an observation schema was used)."""

    frame_timestamp_us: int
    """The frame's capture time (Unix microseconds, as ``Frame.timestamp_us``)."""

    sequence: int
    """The frame's per-participant sequence number (0 when the source does not number frames)."""

    parsed: BaseModel | None = None
    """The observation as an instance of the check's observation schema, when it has one and the reply fit it."""

    confidence: float | None = None
    """The deciding role's confidence in ``label``, when it reports one."""

    reason: str = ""
    """Why the label is ``unknown`` without a decision (an unreadable observation, a failed precondition)."""

    frame: Frame | None = field(default=None, repr=False, compare=False)
    """The frame it judged (its visual evidence; a watch step keeps only its recent judgments)."""


@dataclass(frozen=True, slots=True)
class WatchModels:
    """The models a watch step uses, by what each does."""

    perceive: Role
    """Describes each frame (a vision model), as free text or into the observation schema."""
    decide: Role
    """Labels a description against the criteria (a fast classifier)."""


class CodeJudge(Protocol):
    """Labels one observation in code, in place of the deciding model (:class:`StructuredObservation`).

    ``observation`` is the frame read into the observation schema; ``facts`` the typed facts the step has been given
    (an instance of its ``facts_schema``; ``None`` while it has none); ``inputs`` the step's inputs for this run.
    """

    def __call__(self, observation: Any, facts: Any, inputs: Any) -> Label: ...


@dataclass(frozen=True, slots=True)
class StructuredObservation:
    """Optional for a check: describe each frame in a schema (a structured caption), and optionally judge it in code.

    ``schema`` is what the perceiving model fills in for each frame; shaped well (a gate on what is in view, the
    object named before it is read, values constrained to what is possible: see the step types' README), it is an
    accurate account of the frame. ``code_judge``, when given, labels it in code (``positive``, ``negative`` or
    ``unknown``) in place of the deciding model: for a check that compares values (a reading against a target), where
    code is exact. The step's criteria then only describe the check to its escalator.
    """

    schema: type[BaseModel]
    code_judge: CodeJudge | None = None


@dataclass(frozen=True, slots=True)
class Check:
    """What one run of a watch step verifies, run by :func:`verify`: its words with that run's inputs filled in.

    ``question`` is asked of each frame and ``criteria`` define done. ``facts_schema`` is a schema of facts the check
    may need but the camera may not establish (a display's unit, a vessel's contents); the step's escalator
    (``...escalation``) may ask the person for them. ``observation``, when given, reads each frame into a schema
    (:class:`StructuredObservation`). ``inputs`` are this run's inputs (what a code judge compares against).
    """

    question: str
    criteria: Criteria
    facts_schema: type[BaseModel] | None = None
    observation: StructuredObservation | None = None
    inputs: BaseModel | None = None


@dataclass(frozen=True, slots=True)
class Knowledge:
    """What a step knows beyond the camera: typed ``facts`` (an instance of its ``facts_schema``) and ``notes`` (what
    its escalator learned, as text)."""

    facts: BaseModel | None = None
    notes: tuple[str, ...] = field(default=())

    def text(self) -> str:
        """The knowledge as context for a decision (empty when there is none)."""
        parts = []
        if self.facts is not None:
            values = self.facts.model_dump(exclude_none=True)
            if values:
                parts.append("Known facts (from the person or the app's settings): " + json.dumps(values))
        parts += list(self.notes)
        return "\n".join(parts)


class StepKnowledge:
    """What one run of a watch step knows beyond the camera, as it grows.

    The check reads :attr:`knowledge` on every frame and never writes; the escalator beside the step adds to it
    (:meth:`add`) with what the person answered or said.
    """

    def __init__(self) -> None:
        self.knowledge = Knowledge()

    def add(self, note: str, facts: BaseModel | None = None) -> None:
        """Add a note, and the typed facts it established (merged over earlier ones: a later value replaces)."""
        merged = self.knowledge.facts
        if facts is not None:
            update = facts.model_dump(exclude_none=True)
            merged = facts if merged is None else merged.model_copy(update=update)
        self.knowledge = Knowledge(merged, (*self.knowledge.notes, note))


async def judge(
    frame: Frame,
    check: Check,
    models: WatchModels,
    *,
    knowledge: Knowledge | None = None,
    timeout_s: float | None = None,
) -> Judgment:
    """Describe ``frame`` with ``models.perceive``, then judge the description against ``check``: with the
    observation's ``code_judge`` when it has one, else with ``models.decide``.

    With ``check.observation`` the observation is an instance of its schema; a reply that does not fit it is
    ``unknown`` (reason ``unreadable``) without a decision. The knowledge reaches the judgement, never the
    observation, so what the camera shows is described on its own. ``timeout_s`` bounds the observation call.
    """
    knowledge = knowledge or Knowledge()
    parsed: BaseModel | None = None
    structured = check.observation
    if structured is not None:
        try:
            parsed = await models.perceive.ask(
                frame, check.question, response_format=structured.schema, timeout_s=timeout_s
            )
        except StructuredOutputError:
            return Judgment("unknown", "", frame.timestamp_us, frame.sequence, reason="unreadable", frame=frame)
        description = parsed.model_dump_json(exclude_none=True)
        if structured.code_judge is not None:
            label = structured.code_judge(observation=parsed, facts=knowledge.facts, inputs=check.inputs)
            return Judgment(label, description, frame.timestamp_us, frame.sequence, parsed, frame=frame)
    else:
        description = str(await models.perceive.ask(frame, check.question, timeout_s=timeout_s)).strip()
    context = f"Observation of the current camera frame:\n{description}"
    if known := knowledge.text():
        context += f"\n\n{known}"
    criteria = check.criteria
    decision = await models.decide.decide(criteria.question, choices=criteria.choices(), context=context)
    if str(decision) not in LABELS:
        return Judgment(
            "unknown", description, frame.timestamp_us, frame.sequence, parsed, reason="undecidable reply", frame=frame
        )
    label: Label = str(decision)  # type: ignore[assignment]
    return Judgment(label, description, frame.timestamp_us, frame.sequence, parsed, decision.confidence, frame=frame)
