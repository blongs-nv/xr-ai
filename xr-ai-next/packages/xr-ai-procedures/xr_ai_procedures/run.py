# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The state of one run of a procedure, the facts the user tells it, and how its lines are filled in.

Only the supervisor writes a run after it starts. The controls and the running step only ``post`` commands to its
inbox (``stop``, ``next``, ``back``, ``facts``; ``seen`` and ``outcome`` from the step), and the supervisor applies
them in order.
"""

from __future__ import annotations

import asyncio
import string
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

__all__ = ["Facts", "Run", "fill", "told_model"]


class Facts(BaseModel):
    """Base class for what the user may tell one step: subclass it with optional fields (``None``: not told).

    The model is the extraction contract: give each field a description and the tightest type and bounds you can
    (a ``Literal``, ``ge``/``le``); the schema is sent to the model as the constraint, so the prompt only says the
    task.
    """

    def told(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


def told_model(by_step: Mapping[str, type[BaseModel]]) -> tuple[type[BaseModel], dict[str, str]]:
    """What one utterance can tell the procedure: the facts of every step, plus ``step_done``.

    Every step's facts are read at once, so the user may volunteer something for a step that is ahead. Returns the
    model and the step owning each fact. Field-level constraints carry over; model-level validators do not. Two
    steps declaring the same fact name is an error.
    """
    fields: dict[str, Any] = {}
    owner: dict[str, str] = {}
    for step_id, model in by_step.items():
        for name, info in model.model_fields.items():
            if name in owner or name == "step_done":
                raise ValueError(f"fact {name!r} is declared by both {owner.get(name, 'the procedure')} and {step_id}")
            owner[name] = step_id
            fields[name] = (info.annotation | None, info)
    fields["step_done"] = (
        bool | None,
        Field(
            None,
            description=(
                "True if the user says the current step is already done or the thing the procedure asked about is "
                'so ("yes" to a yes/no question); false if they say it is not; otherwise null.'
            ),
        ),
    )
    return create_model("Told", **fields), owner


@dataclass(frozen=True, slots=True)
class Failed:
    """A step that raised: its error, as the step's outcome."""

    error: str


class Stopped(Exception):
    """Raised inside the supervisor when the user stops the run."""


async def cancel(task: asyncio.Task[Any] | None) -> None:
    """Stop a step and wait until it has really finished."""
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@dataclass
class Run:
    """One run of a procedure for one connection (``p.session.state``)."""

    facts: dict[str, BaseModel]
    """What the user told each step, by step id (only steps that declare facts)."""
    status: Literal["running", "completed", "failed", "stopped"] = "running"
    index: int = 0
    """The current step."""
    results: dict[str, Any] = field(default_factory=dict)
    """Step id to the value its ``Done`` carried."""
    outcomes: dict[str, str] = field(default_factory=dict)
    """Step id to how it ended: ``done``, ``confirmed`` (the user said so) or ``skipped``."""
    seen: str = ""
    """What the running step last reported."""
    pending: str | None = None
    """What the procedure asked the user and waits for (``confirm``, ``view``, ``retry`` or a fact)."""
    asked: str = ""
    """The words of that question."""
    carry: str = ""
    """Said together with the next utterance (a step's ``done`` line, before the next step's ``enter``)."""
    generation: int = 0
    """Counts step builds; a step's posts carry it, so a replaced step's late posts are ignored."""
    inbox: deque[tuple[str, Any]] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)

    def told(self) -> dict[str, Any]:
        """Everything the user has told the procedure, across steps."""
        return {k: v for m in self.facts.values() for k, v in m.model_dump(exclude_none=True).items()}

    def post(self, command: str, payload: Any = None) -> None:
        """Hand the supervisor a command (the only way anything but the supervisor changes a run)."""
        self.inbox.append((command, payload))
        self.wake.set()


class _Keep(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def fill(text: str, values: Mapping[str, Any], *args: Any) -> str:
    """``text`` with its ``{name}`` placeholders filled from ``values`` (and ``{}`` from ``args``).

    A placeholder with no value is left as written; a line that cannot be formatted is returned unchanged.
    """
    try:
        return string.Formatter().vformat(text, args, _Keep(values))
    except (ValueError, IndexError, KeyError, AttributeError, TypeError):
        return text
