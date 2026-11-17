# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The link between one run of a step and its escalator: the only thing the two share.

Two contexts, one each way:

- **outbound** (step to escalator): what the step shows about itself while it runs, :class:`Outbound`: what it does
  and when it cannot decide, its recent reports, what it knows, and the typed facts it accepts. The step attaches a
  function that builds it from its live state, so the escalator always reads the current one.
- **inbound** (escalator to step): what the escalator learned, :class:`Inbound`: a note and, optionally, typed facts
  (an instance of the outbound ``needs``). The step attaches a function that adds it to what it reads, so it takes
  effect on the step's next decision.

A step type supports escalation by implementing :class:`Escalatable` (``linked(link)``: a copy of the step that
attaches its two ends when it runs). A :class:`Link` is made for each run, never shared between runs.

The escalator talks to the person on its own, through :class:`EscalationIO`; it never speaks through the step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..steps.step_api import Step


@dataclass(frozen=True, slots=True)
class Report:
    """One of a step's recent reports, as its escalator reads it."""

    label: str
    """The step's own label for it (a watch step's ``positive``, ``negative``, ``unknown``)."""
    text: str
    """What the step observed, as text."""
    decided: bool
    """Whether the step could decide from it (an undecided one is what an escalator watches for)."""


@dataclass(frozen=True, slots=True)
class Outbound:
    """What a step shows its escalator about itself, as of now."""

    step_id: str
    started_s: float
    """When the step started (core clock)."""
    task: tuple[str, ...]
    """Lines saying what the step does, when it completes and when it cannot decide."""
    reports: tuple[Report, ...]
    """Its recent reports, oldest first."""
    known: str = ""
    """What it knows beyond its own observations (the app's settings, what the person said); empty when nothing."""
    needs: type[BaseModel] | None = None
    """The typed facts it accepts inbound (``None``: notes only)."""


@dataclass(frozen=True, slots=True)
class Inbound:
    """What an escalator gives a step: a note, and the typed facts it established (an instance of ``needs``)."""

    note: str
    known: BaseModel | None = None


class Link:
    """One run's link between a step and its escalator; see the module docstring."""

    def __init__(self) -> None:
        self._outbound: Callable[[], Outbound] | None = None
        self._inbound: Callable[[Inbound], None] | None = None

    # The step's side.
    def attach(self, outbound: Callable[[], Outbound], inbound: Callable[[Inbound], None]) -> None:
        """Called by the step as it starts: how to build its outbound context, and where inbound context goes."""
        self._outbound, self._inbound = outbound, inbound

    # The escalator's side.
    def outbound(self) -> Outbound | None:
        """The step's outbound context now (``None`` until the step has started)."""
        return self._outbound() if self._outbound is not None else None

    def inbound(self, context: Inbound) -> None:
        """Give the step ``context``; it reads it from its next decision on."""
        if self._inbound is None:
            raise RuntimeError("the step has not started; it takes inbound context once it runs")
        self._inbound(context)


@runtime_checkable
class Escalatable(Protocol):
    """A step type that supports escalation."""

    def linked(self, link: Link) -> Step[Any]:
        """A copy of the step that, when it runs, attaches its outbound and inbound context to ``link``."""
        ...


class EscalationIO(Protocol):
    """What an escalator touches: the person, and what is going on around the step. Live it is the procedure's
    :class:`~..runner.Run`; in an evaluation, the case's scripted person."""

    async def say(self, text: str) -> None:
        """Tell the person something."""
        ...

    async def ask(self, question: str) -> str | None:
        """Ask the person and wait for the answer; ``None`` when none came."""
        ...

    async def situation(self, sources: Sequence[str]) -> str:
        """What is going on around the step, from ``sources`` in order (``run``: the run's summary; context
        channels by name, such as ``dialogue``); empty when they have nothing."""
        ...
