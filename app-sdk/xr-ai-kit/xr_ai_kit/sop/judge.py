# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One camera frame judged against a step's criteria: the unit every visual check is built from.

An observing role describes the frame (free text, or an instance of a schema); a deciding role reads that
description and picks ``positive``, ``negative`` or ``unknown`` against the criteria. Keeping the two apart means the
observation can be read back and evaluated on its own, and the same function runs live (inside :func:`verify`) and
offline (an evaluation over recorded frames).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import BaseModel
from xr_ai_app import Frame, Role, StructuredOutputError

Label = Literal["positive", "negative", "unknown"]
LABELS: tuple[Label, ...] = ("positive", "negative", "unknown")

_Observed = TypeVar("_Observed", bound=BaseModel)


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


async def judge(
    frame: Frame,
    *,
    observe: Role,
    decide: Role,
    question: str,
    criteria: Criteria,
    observation_model: type[_Observed] | None = None,
    can_assess: Callable[[_Observed], bool] | None = None,
    timeout_s: float | None = None,
) -> Judgment:
    """Describe ``frame`` with ``observe``, then judge the description against ``criteria`` with ``decide``.

    With ``observation_model`` the observation is an instance of that schema; a reply that does not fit it is
    ``unknown`` (reason ``unreadable``) without a decision. ``can_assess`` is a check in code over that instance
    (for example "a temperature and its unit were read"): when it is false the frame is ``unknown`` (reason
    ``not assessable``) and the deciding role is not asked. ``timeout_s`` bounds the observation call.
    """
    parsed: BaseModel | None = None
    if observation_model is not None:
        try:
            parsed = await observe.ask(frame, question, response_format=observation_model, timeout_s=timeout_s)
        except StructuredOutputError:
            return Judgment("unknown", "", frame.timestamp_us, frame.sequence, reason="unreadable")
        description = parsed.model_dump_json(exclude_none=True)
        if can_assess is not None and not can_assess(parsed):  # type: ignore[arg-type]
            return Judgment("unknown", description, frame.timestamp_us, frame.sequence, parsed, reason="not assessable")
    else:
        description = str(await observe.ask(frame, question, timeout_s=timeout_s)).strip()
    context = f"Observation of the current camera frame:\n{description}"
    decision = await decide.decide(criteria.instructions, choices=criteria.choices(), context=context)
    label: Label = str(decision) if str(decision) in LABELS else "unknown"  # type: ignore[assignment]
    return Judgment(label, description, frame.timestamp_us, frame.sequence, parsed, decision.confidence)
