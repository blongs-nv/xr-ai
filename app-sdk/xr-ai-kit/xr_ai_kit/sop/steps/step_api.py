# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a step is and what it may touch: the contract every step type, the runner and the evaluation share.

A step (:class:`Step`) is an object with an ``id``, the line it speaks as it starts (``say``) and ``run(io)``. It
touches the world only through ``io`` (:class:`StepIO`): the camera frames, speaking, asking the person, reporting
its progress, and what is going on around it. Live, ``io`` is the procedure's :class:`~..runner.Run`; in an
evaluation it is a scripted one (:mod:`..evaluation`) that plays recorded frames and a scripted person, so the
evaluation runs exactly the code that ships.

How a case of each step type is scored lives in that type's own scoring module (``watch/watch_scoring.py``, ...);
the steps never import evaluation code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from xr_ai_app import Frame, Role

T_co = TypeVar("T_co", covariant=True)


class Frames(Protocol):
    """The camera frames a step reads (``Participant.frames`` live, a recorded scene in an evaluation)."""

    def every(self, interval_s: float, *, timeout_s: float = 5.0) -> AsyncIterator[Frame]: ...

    async def current(self, *, max_age_s: float = 2.0, timeout_s: float = 5.0) -> Frame | None: ...


class StepIO(Protocol):
    """Everything a step may touch. :class:`~..runner.Run` is one; :mod:`..evaluation` has a scripted one."""

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

    async def situation(self, sources: Sequence[str]) -> str:
        """What is going on around the step, from ``sources`` in order: ``run`` (the run's summary) and context
        channels by name (``dialogue``); empty when they have nothing."""
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
    in text (deriving settings, reading an answer)."""

    observe: Role
    decide: Role
    reason: Role
