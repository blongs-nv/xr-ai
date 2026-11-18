# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The channel between one run of a step and its escalator: the only thing the two share.

It carries one thing each way:

- **a snapshot of the step** (step to escalator), :class:`StepSnapshot`: what the step does and when it cannot decide,
  its recent :class:`Evidence`, what it knows, and the schema of facts it accepts. The step connects a function that
  builds it from its live state, so the escalator always reads the current one.
- **added context** (escalator to step), :class:`AddedContext`: what the escalator learned, as a note and, optionally,
  typed facts (an instance of the snapshot's ``facts_schema``). The step connects a function that adds it to what it
  decides with, so it takes effect on the step's next decision.

A step type supports escalation by implementing :class:`SupportsEscalation`: ``run_with_escalation(environment,
channel, inputs)``, which runs the step with its end of ``channel`` connected. A channel is made for each run, never
shared between runs.

An escalator is anything with ``run(channel, environment)`` (:class:`Escalator`): the kit's reasons in text over the
step's evidence (:class:`~.reasoning_escalator.ReasoningEscalator`); another may look at the evidence itself, the
frame behind each piece (:attr:`Evidence.frame`) or the camera now (``environment.frames``), with a model of its own.
It talks to the person on its own, through its :class:`EscalatorEnvironment`; it never speaks through the step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel
from xr_ai_app import Frame

from ..steps.step_api import Frames, StepEnvironment


@dataclass(frozen=True, slots=True)
class Evidence:
    """One of the step's recent attempts to decide, as its escalator reads it."""

    label: str
    """The step's own label for it (a watch step's ``positive``, ``negative``, ``unknown``)."""
    description: str
    """What the step observed, as text."""
    decided: bool
    """Whether the step could decide from it (an undecided one is what an escalator watches for)."""
    frame: Frame | None = None
    """The camera frame it was made from, when the step keeps it."""


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    """What a step shows its escalator about itself, as of now."""

    step_id: str
    started_s: float
    """When the step started (core clock)."""
    task: tuple[str, ...]
    """Lines saying what the step does, when it completes and when it cannot decide."""
    evidence: tuple[Evidence, ...]
    """Its recent attempts to decide, oldest first."""
    known: str = ""
    """What it knows beyond its own observations (the app's settings, what the person said); empty when nothing."""
    facts_schema: type[BaseModel] | None = None
    """The typed facts it accepts as added context (``None``: notes only)."""


@dataclass(frozen=True, slots=True)
class AddedContext:
    """What an escalator gives a step: a note, and the typed facts it established (an instance of the snapshot's
    ``facts_schema``)."""

    note: str
    facts: BaseModel | None = None


class EscalationChannel:
    """One run's channel between a step and its escalator; see the module docstring."""

    def __init__(self) -> None:
        self._snapshot: Callable[[], StepSnapshot] | None = None
        self._add_context: Callable[[AddedContext], None] | None = None

    # The step's end.
    def connect_step(self, snapshot: Callable[[], StepSnapshot], add_context: Callable[[AddedContext], None]) -> None:
        """Called by the step as it starts: how to build its snapshot, and how to add context to it."""
        self._snapshot, self._add_context = snapshot, add_context

    # The escalator's end.
    def snapshot(self) -> StepSnapshot | None:
        """The step as of now (``None`` until the step has started)."""
        return self._snapshot() if self._snapshot is not None else None

    def add_context(self, context: AddedContext) -> None:
        """Give the step ``context``; it decides with it from its next decision on."""
        if self._add_context is None:
            raise RuntimeError("the step has not started; it takes added context once it runs")
        self._add_context(context)


ESCALATOR_SPEAKING: ContextVar[bool] = ContextVar("escalator_speaking", default=False)
"""True inside an escalator's own task (:class:`~.step_with_escalator.StepWithEscalator` sets it there): what is said
or asked from that task is the escalator's, not the step's."""


def speaker() -> Literal["step", "escalator"]:
    """Who is speaking from the current task: an escalator beside a step, or the step (or the procedure) itself. The
    run's record and an evaluation's trace tag each line with it."""
    return "escalator" if ESCALATOR_SPEAKING.get() else "step"


@runtime_checkable
class SupportsEscalation(Protocol):
    """A step type that supports escalation."""

    async def run_with_escalation(
        self, environment: StepEnvironment, channel: EscalationChannel, inputs: Any = None
    ) -> Any:
        """Run the step (as ``run`` does) with its end of ``channel`` connected."""
        ...


@runtime_checkable
class EscalatorEnvironment(Protocol):
    """What an escalator touches: the person, the camera, and what is going on around the step. Live it is the
    procedure's :class:`~..runner.Run`; in an evaluation, the case's scripted person and scene."""

    @property
    def frames(self) -> Frames:
        """The camera now (the same frames the step sees)."""
        ...

    async def say(self, text: str) -> None:
        """Tell the person something."""
        ...

    async def ask(self, question: str) -> str | None:
        """Ask the person and wait for the answer; ``None`` when none came."""
        ...

    async def read_context(self, sources: Sequence[str]) -> str:
        """What is going on around the step, from ``sources`` in order (``run``: the run's summary; context
        channels by name, such as ``dialogue``); empty when they have nothing."""
        ...


@runtime_checkable
class Escalator(Protocol):
    """An agent beside one step: runs until cancelled, reading the step through ``channel`` and reaching the person
    through ``environment``. The kit's is :class:`~.reasoning_escalator.ReasoningEscalator`; an app may bring its
    own."""

    async def run(self, channel: EscalationChannel, environment: EscalatorEnvironment) -> None: ...
