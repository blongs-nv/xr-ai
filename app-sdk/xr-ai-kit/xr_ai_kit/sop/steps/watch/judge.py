# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One camera frame judged against a check: the unit every visual check is built from.

A :class:`Check` declares what a step verifies: the question asked of each frame, the criteria for done, and,
optionally, what it may need to know that the camera cannot show (``needs``; a step's escalator may ask the person
for it). An observing role describes the frame (free text, or an instance of a schema); a deciding role
reads that description, with the :class:`Facts` known so far, and picks ``positive``, ``negative`` or ``unknown``
against the criteria. Keeping the two apart means the observation can be read back and evaluated on its own, and
the same function runs live (inside :func:`verify`) and offline (an evaluation over recorded frames).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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

    def format(self, **values: object) -> Criteria:
        """A copy with ``{name}`` placeholders filled in every field (for targets known only at run time)."""
        return Criteria(
            self.instructions.format(**values),
            self.positive.format(**values),
            self.negative.format(**values),
            self.unknown.format(**values),
        )


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
    """The observation as an instance of the schema, when one was given and the reply fit it."""

    confidence: float | None = None
    """The deciding role's confidence in ``label``, when it reports one."""

    reason: str = ""
    """Why the label is ``unknown`` without a decision (an unreadable observation, a failed precondition)."""


@dataclass(frozen=True, slots=True)
class Check:
    """What one visual check verifies; declared by the app, run by :func:`verify`.

    ``question`` is asked of each frame and ``criteria`` define done. ``observation_model`` makes the observation a
    schema instance, and ``can_assess(observation, known)`` is a check in code over it and the typed facts known so
    far (``known`` is ``None`` when the check has no ``needs``): when it is false the frame is ``unknown`` without a
    decision. ``reconcile(observation, known)`` returns the observation corrected with the known facts before anything
    judges it (a unit the person stated replacing one the model claims to read): known facts win in code, not by
    asking the deciding model to prefer them. ``needs`` is a schema of facts the check may need but the camera may
    not establish (a display's unit, a vessel's contents); the step's escalator (:mod:`.escalator`) may ask the person
    for them.
    """

    question: str
    criteria: Criteria
    needs: type[BaseModel] | None = None
    observation_model: type[BaseModel] | None = None
    can_assess: Callable[[Any, Any], bool] | None = None
    reconcile: Callable[[Any, Any], Any] | None = None

    def format(self, **values: object) -> Check:
        """A copy with ``{name}`` placeholders filled in the criteria (targets known only at run time)."""
        return replace(self, criteria=self.criteria.format(**values))


@dataclass(frozen=True, slots=True)
class Facts:
    """What is known beyond the camera: typed facts (an instance of a check's ``needs``) and what the person said."""

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
    """A step's external context channel: what is known from outside the fast loop.

    The check reads :attr:`facts` on every frame and never writes; anything outside the loop (an app setting at the
    start, the side escalation with what the person answered) adds to it with :meth:`add`. One per step run.
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
    *,
    observe: Role,
    decide: Role,
    facts: Facts | None = None,
    timeout_s: float | None = None,
) -> Judgment:
    """Describe ``frame`` with ``observe``, then judge the description against ``check`` with ``decide``.

    With ``check.observation_model`` the observation is an instance of that schema; a reply that does not fit it is
    ``unknown`` (reason ``unreadable``) without a decision. When ``check.can_assess`` is false for the observation and
    the known facts, the frame is ``unknown`` (reason ``not assessable``) and the deciding role is not asked. The
    facts reach the decision, never the observation, so what the camera shows is described on its own. ``timeout_s``
    bounds the observation call.
    """
    facts = facts or Facts()
    parsed: BaseModel | None = None
    if check.observation_model is not None:
        try:
            parsed = await observe.ask(
                frame, check.question, response_format=check.observation_model, timeout_s=timeout_s
            )
        except StructuredOutputError:
            return Judgment("unknown", "", frame.timestamp_us, frame.sequence, reason="unreadable")
        if check.reconcile is not None:
            parsed = check.reconcile(parsed, facts.known)
        description = parsed.model_dump_json(exclude_none=True)
        if check.can_assess is not None and not check.can_assess(parsed, facts.known):
            return Judgment("unknown", description, frame.timestamp_us, frame.sequence, parsed, reason="not assessable")
    else:
        description = str(await observe.ask(frame, check.question, timeout_s=timeout_s)).strip()
    context = f"Observation of the current camera frame:\n{description}"
    if known := facts.text():
        context += f"\n\n{known}"
    criteria = check.criteria
    decision = await decide.decide(criteria.instructions, choices=criteria.choices(), context=context)
    label: Label = str(decision) if str(decision) in LABELS else "unknown"  # type: ignore[assignment]
    return Judgment(label, description, frame.timestamp_us, frame.sequence, parsed, decision.confidence)
