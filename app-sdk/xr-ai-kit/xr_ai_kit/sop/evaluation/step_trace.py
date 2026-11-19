# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What an evaluation records of one run of a step (:class:`StepTrace`), which a step type's scoring reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Line:
    """Something said to the person while a step ran."""

    t_s: float
    """When, on the evaluation's scene clock (the recording's own time for a scene that is one part of it)."""

    kind: Literal["ask", "say", "done"]
    """``ask`` and ``say`` during the step; ``done``: its ``done_line``."""
    text: str
    answer: str | None = None
    """For ``ask``: the person's answer (``None``: none came)."""
    by: str = "step"
    """Who said it: the step itself, or the escalator beside it."""


@dataclass(slots=True)
class StepTrace:
    """What one run of a step did, as an evaluation records it; what a step type's ``score`` reads."""

    step: str
    completed: bool = False
    """Whether ``run`` returned (the step finished) before the scene ended."""

    completed_at_s: float | None = None
    result: Any = None
    """What ``run`` returned."""

    lines: list[Line] = field(default_factory=list)
    states: list[tuple[float, Any]] = field(default_factory=list)
    """Every reported state, with its time on the scene clock."""

    error: str | None = None

    def asked(self, by: str | None = None) -> list[Line]:
        """The questions asked (by ``by`` only, when given)."""
        return [line for line in self.lines if line.kind == "ask" and by in (None, line.by)]

    def said(self, by: str | None = None) -> list[Line]:
        """The things told (by ``by`` only, when given)."""
        return [line for line in self.lines if line.kind == "say" and by in (None, line.by)]


def mentions(text: str, words: list[str]) -> bool:
    """Whether ``text`` contains any of ``words`` (case-insensitive); an empty list matches anything."""
    lowered = text.casefold()
    return not words or any(word.casefold() in lowered for word in words)
