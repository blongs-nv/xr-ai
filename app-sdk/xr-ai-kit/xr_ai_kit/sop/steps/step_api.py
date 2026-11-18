# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a step is and what it may touch: the contract every step type, the runner and the evaluation share.

A step (:class:`Step`) has an ``id``, the line spoken as it starts (``start_line``), the line spoken when it finishes
(``done_line(result)``) and ``run(environment)``. It touches the world only through its
:class:`StepEnvironment`: the camera, speaking to the person, asking them something, and reporting its progress.
Live, the environment is the procedure's :class:`~..runner.Run`; in an evaluation it is a scripted one
(:mod:`..evaluation`) that plays recorded frames, so the evaluation runs exactly the code that ships.

A step type (the kit's ``watch``, ``read``, ``wait``, or an app's own) is any class with that shape. The kit's types
are built in one call from the step's definition (:class:`~.step_definitions.StepDefinition`, its words from the
procedure's YAML), the models it uses, and ``values`` for the ``{placeholders}`` in its words (targets known only at
run time); :func:`fill_placeholders` fills them. A step type may also have ``scoring``, naming the module that scores
its evaluation cases (``Expect`` and ``score(trace, expect)``), imported only by the evaluation.

Escalation (an agent beside the step that helps the person when the step cannot decide) is not part of this
contract: a step type opts into it separately (``..escalation``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol, TypeVar, runtime_checkable

from xr_ai_app import Frame

T_co = TypeVar("T_co", covariant=True)


class Frames(Protocol):
    """The camera frames a step reads (``Participant.frames`` live, a recorded scene in an evaluation)."""

    def every(self, interval_s: float, *, timeout_s: float = 5.0) -> AsyncIterator[Frame]: ...

    async def current(self, *, max_age_s: float = 2.0, timeout_s: float = 5.0) -> Frame | None: ...


class StepEnvironment(Protocol):
    """Everything a step may touch. :class:`~..runner.Run` is one; :mod:`..evaluation` has a scripted one."""

    @property
    def frames(self) -> Frames:
        """The person's camera."""
        ...

    async def say(self, text: str) -> None:
        """Tell the person something."""
        ...

    async def ask(self, question: str) -> str | None:
        """Ask the person something and wait for the answer; ``None`` when none came."""
        ...

    async def report_progress(self, step_id: str, state: Any) -> None:
        """Record the step's latest state (its ``summary()``, if it has one), for the conversation and the record."""
        ...


@runtime_checkable
class Step(Protocol[T_co]):
    """One step of a procedure: ``await run.step(step)`` live, a case file in an evaluation."""

    id: str
    start_line: str | None
    """Spoken as the step starts (``None``: nothing)."""

    def done_line(self, result: Any) -> str | None:
        """Spoken when the step finishes with ``result`` (``None``: nothing); a skipped step says nothing."""
        ...

    async def run(self, environment: StepEnvironment) -> T_co: ...


def fill_placeholders(template: str | None, values: Mapping[str, object]) -> str | None:
    """``template`` with its ``{name}`` placeholders filled from ``values`` (``None`` stays ``None``)."""
    return None if template is None else template.format(**values)
