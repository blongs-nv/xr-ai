# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a step is and what it may touch: the contract every step type, the runner and the evaluation share.

A step (:class:`Step`) is an object with an ``id``, the line spoken as it starts (``say``) and ``run(io)``. It
touches the world only through ``io`` (:class:`StepIO`): the camera frames, speaking, and reporting its progress.
Live, ``io`` is the procedure's :class:`~..runner.Run`; in an evaluation it is a scripted one (:mod:`..evaluation`)
that plays recorded frames, so the evaluation runs exactly the code that ships.

A step type (the kit's ``watch``, ``read``, ``wait``, or an app's own) is any class with that shape. Optionally it
also has:

- ``closing(result) -> str | None``: what is spoken when it finishes, from its result (the runner speaks it, as it
  speaks ``say`` when the step starts; a skipped step says nothing);
- its models as fields of its own, named for what each does (``WatchStep(observe=..., decide=...)``): every type
  needs different ones, so nothing here names a model;
- ``define(definition, ...)``, building the step from its entry in the procedure's YAML
  (:class:`~.step_definitions.StepDefinition`), reading its own section of it;
- ``scoring``, naming the module that scores its evaluation cases (``Expect`` and ``score(trace, expect)``),
  imported only by the evaluation.

Anything else a step type supports is its own: escalation, for one, is a separate package (``..escalation``) that a
step type opts into.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from xr_ai_app import Frame

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
        """Speak on the assistant's own initiative."""
        ...

    def report(self, step_id: str | None = None) -> Callable[[Any], Awaitable[None]]:
        """An ``on_state`` callback for the step's progress."""
        ...


@runtime_checkable
class Step(Protocol[T_co]):
    """One step of a procedure: ``await run.step(step)`` live, a case file in an evaluation."""

    id: str
    say: str | None

    async def run(self, io: StepIO) -> T_co: ...


def closing_line(step: object, result: Any) -> str | None:
    """What ``step`` says when it finishes with ``result`` (its ``closing``, if it has one)."""
    closing = getattr(step, "closing", None)
    return closing(result) if callable(closing) else None
