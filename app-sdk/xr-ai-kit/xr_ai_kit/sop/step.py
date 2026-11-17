# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a step is, what it may touch, and what an evaluation records of it.

A step (:class:`Step`) is an object with an ``id``, the line it speaks as it starts (``say``) and ``run(io)``. It
touches the world only through ``io`` (:class:`StepIO`): the camera frames, speaking, asking the person, reporting
its progress, and what is going on around it. Live, ``io`` is the procedure's :class:`~.runner.Run`; in an
evaluation it is a scripted one (:mod:`.eval`) that plays recorded frames and a scripted person, so the evaluation
runs exactly the code that ships.

Each step type (:mod:`.steps`) also says how a case of it is scored: ``Expect`` (what a case may expect of a step of
that type) and ``score(trace, expect)``, next to its ``run``. The evaluation never looks inside a step; it records a
:class:`StepTrace` and hands it to the step's own ``score``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from xr_ai_app import Frame, Role

T_co = TypeVar("T_co", covariant=True)


class Frames(Protocol):
    """The camera frames a step reads (``Participant.frames`` live, a recorded scene in an evaluation)."""

    def every(self, interval_s: float, *, timeout_s: float = 5.0) -> AsyncIterator[Frame]: ...

    async def current(self, *, max_age_s: float = 2.0, timeout_s: float = 5.0) -> Frame | None: ...


class StepIO(Protocol):
    """Everything a step may touch. :class:`~.runner.Run` is one; :mod:`.eval` has a scripted one."""

    @property
    def frames(self) -> Frames: ...

    async def say(self, text: str) -> None:
        """Speak on the assistant's own initiative (an instruction, a nudge)."""
        ...

    async def ask(self, question: str) -> str | None:
        """Ask the person and wait for the answer; ``None`` when none came."""
        ...

    def report(self, step_id: str | None = None) -> Callable[[Any], Awaitable[None]]:
        """An ``on_state`` callback for the step's progress."""
        ...

    async def situation(self) -> str:
        """What is going on around the step (the procedure's state, the recent conversation); may be empty."""
        ...


@runtime_checkable
class Step(Protocol[T_co]):
    """One step of a procedure: ``await run.step(step)`` live, a case file in an evaluation."""

    id: str
    say: str | None

    async def run(self, io: StepIO) -> T_co: ...


@dataclass(frozen=True, slots=True)
class Roles:
    """The models the steps use: ``observe`` looks at a frame, ``decide`` picks a label (fast), ``reason`` reasons
    in text (deriving settings, reading an answer, the side escalation)."""

    observe: Role
    decide: Role
    reason: Role


@dataclass(frozen=True, slots=True)
class Line:
    """Something said to the person while a step ran."""

    t_s: float
    """When, on the evaluation's scene clock (the recording's own time for a scene that is one part of it)."""

    kind: Literal["ask", "say"]
    text: str
    answer: str | None = None
    """For ``ask``: the person's answer (``None``: none came)."""


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

    def asked(self) -> list[Line]:
        return [line for line in self.lines if line.kind == "ask"]

    def said(self) -> list[Line]:
        return [line for line in self.lines if line.kind == "say"]


def mentions(text: str, words: list[str]) -> bool:
    """Whether ``text`` contains any of ``words`` (case-insensitive); an empty list matches anything."""
    lowered = text.casefold()
    return not words or any(word.casefold() in lowered for word in words)
