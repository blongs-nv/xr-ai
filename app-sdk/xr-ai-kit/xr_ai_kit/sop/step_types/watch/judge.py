# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One camera frame judged against a check: the unit every visual check is built from.

A :class:`Check` declares what a step verifies: the question asked of each frame, the criteria for done, and,
optionally, the typed facts it may need that the camera cannot show (``facts_schema``; a step's escalator may ask the
person for them) and how to read each frame into a schema (:class:`StructuredObservation`). The perceiving model
(:class:`WatchModels`) describes the frame (free text, or an instance of the schema); the deciding model reads that
description, with the :class:`Facts` known so far, and picks ``positive``, ``negative`` or ``unknown`` against the
criteria. Keeping the two apart means the observation can be read back and evaluated on its own, and
the same function runs live (inside :func:`verify`) and offline (an evaluation over recorded frames).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel
from xr_ai_app import Frame, Role, StructuredOutputError

Label = Literal["positive", "negative", "unknown"]
LABELS: tuple[Label, ...] = ("positive", "negative", "unknown")


@dataclass(frozen=True, slots=True)
class Criteria:
    """What counts as done for one check: the question the deciding role answers and one criterion per label."""

    instructions: str
    """The question the deciding role answers about an observation (for example "Is there water in the kettle?")."""

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

    frame_us: int
    """The frame's capture time (Unix microseconds)."""

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


@dataclass(frozen=True, slots=True)
class StructuredObservation:
    """Optional for a check: describe each frame in a schema (a structured caption), and optionally judge it in code.

    ``schema`` is what the perceiving model fills in for each frame; shaped well (a gate on what is in view, the
    object named before it is read, values constrained to what is possible: see the step types' README), it is an
    accurate account of the frame. ``judge(observation, facts, inputs)``, when given, labels it in code (``positive``,
    ``negative`` or ``unknown``) in place of the deciding model: for a check that compares values (a reading against a
    target), where code is exact. ``facts`` is the typed facts known so far (``None`` while none are), ``inputs`` the
    step's inputs for this run.
    """

    schema: type[BaseModel]
    judge: Callable[[Any, Any, Any], Label] | None = None


@dataclass(frozen=True, slots=True)
class Check:
    """What one visual check verifies; run by :func:`verify`.

    ``question`` is asked of each frame and ``criteria`` define done. ``facts_schema`` is a schema of facts the check
    may need but the camera may not establish (a display's unit, a vessel's contents); the step's escalator
    (``...escalation``) may ask the person for them. ``observation``, when given, reads each frame into a schema
    (:class:`StructuredObservation`).
    """

    question: str
    criteria: Criteria
    facts_schema: type[BaseModel] | None = None
    observation: StructuredObservation | None = None
    inputs: BaseModel | None = None
    """The step's inputs for this run (what a code judge compares against)."""


@dataclass(frozen=True, slots=True)
class Facts:
    """What is known beyond the camera: typed facts (an instance of a check's ``facts_schema``) and notes (what the
    person said)."""

    known: BaseModel | None = None
    said: tuple[str, ...] = field(default=())

    def text(self) -> str:
        """The facts as context for a decision (empty when there are none)."""
        parts = []
        if self.known is not None:
            values = self.known.model_dump(exclude_none=True)
            if values:
                parts.append("Known facts (from the person or the app's settings): " + json.dumps(values))
        parts += list(self.said)
        return "\n".join(parts)

    def with_answer(self, line: str, known: BaseModel | None = None) -> Facts:
        """These facts plus one exchange with the person, and the typed facts it established."""
        merged = self.known
        if known is not None:
            update = known.model_dump(exclude_none=True)
            merged = known if merged is None else merged.model_copy(update=update)
        return Facts(merged, (*self.said, line))


class StepContext:
    """What a watch step knows from outside its fast loop, for one run of the step.

    The check reads :attr:`facts` on every frame and never writes; anything outside the loop (an app setting at the
    start, the escalator beside the step with what the person answered) adds to it with :meth:`add`.
    """

    def __init__(self, known: BaseModel | None = None) -> None:
        self._facts = Facts(known)

    @property
    def facts(self) -> Facts:
        return self._facts

    def add(self, line: str, known: BaseModel | None = None) -> None:
        """Add one piece of context (and the typed facts it established, if any)."""
        self._facts = self._facts.with_answer(line, known)


async def judge(
    frame: Frame,
    check: Check,
    models: WatchModels,
    *,
    facts: Facts | None = None,
    timeout_s: float | None = None,
) -> Judgment:
    """Describe ``frame`` with ``models.perceive``, then judge the description against ``check``: with the
    observation's ``judge`` in code when it has one, else with ``models.decide``.

    With ``check.observation`` the observation is an instance of its schema; a reply that does not fit it is
    ``unknown`` (reason ``unreadable``) without a decision. The facts reach the judgement, never the observation, so
    what the camera shows is described on its own. ``timeout_s`` bounds the observation call.
    """
    facts = facts or Facts()
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
        if structured.judge is not None:
            label = structured.judge(parsed, facts.known, check.inputs)
            return Judgment(label, description, frame.timestamp_us, frame.sequence, parsed, frame=frame)
    else:
        description = str(await models.perceive.ask(frame, check.question, timeout_s=timeout_s)).strip()
    context = f"Observation of the current camera frame:\n{description}"
    if known := facts.text():
        context += f"\n\n{known}"
    criteria = check.criteria
    decision = await models.decide.decide(criteria.instructions, choices=criteria.choices(), context=context)
    label: Label = str(decision) if str(decision) in LABELS else "unknown"  # type: ignore[assignment]
    return Judgment(label, description, frame.timestamp_us, frame.sequence, parsed, decision.confidence, frame=frame)
