# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A procedure's steps written in one YAML file (:class:`StepDefinitions`), whatever their types.

Only words live in the file; the step's type, models, schemas and code hooks are chosen in code, where the step is
built from its definition (``WatchStep.define(DEFINITIONS["heat_water"], observe=..., decide=...)``). One entry per
step, keyed by its id, and ``lines`` for what the procedure says that belongs to no one step::

    fill_water:
      does: fill the kettle with fresh water                    # every step: the plan the conversation reads
      say: Fill a kettle with fresh water and keep it in view.  # every step: spoken as it starts
      done: Water is visible in the vessel.                     # every step: spoken when it finishes
      watch:                                                    # a section: the watch step's own words
        question: Describe whether water is visibly inside the kettle ...
        criteria: {instructions: ..., positive: ..., negative: ..., unknown: ...}
      escalation: The kettle has to be in the camera's view ... # a section: the step's escalator's guidance
    lines:
      not_identified: I'll use the default settings.

This module knows only ``does``, ``say`` and ``done``. Every other key of an entry is a section, owned by whatever
reads it (a step type, an escalator, an app's own component) with :meth:`StepDefinition.section`, which validates it
against the reader's own type (a words model with ``extra="forbid"``, or ``str``), so nothing here changes for a new
step type. :meth:`StepDefinitions.unread` lists the sections nothing has read (a misspelled section name, or one
whose reader is not wired), for an app to check once it has built its steps.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import TypeAdapter, ValidationError

T = TypeVar("T")

SHARED = ("does", "say", "done")
"""The keys every step's entry may have; every other key is a section."""


@dataclass(frozen=True, slots=True)
class StepDefinition:
    """One step's entry: what every step has (``does``, ``say``, ``done``) and its sections (:meth:`section`)."""

    id: str
    does: str | None = None
    """What the person does in this step, for the plan the conversation reads."""
    say: str | None = None
    """Spoken as the step starts."""
    done: str | None = None
    """Spoken when the step finishes ({placeholders} filled from its result; see the step's type)."""
    sections: Mapping[str, Any] = field(default_factory=dict)
    """The entry's other keys, each read by its owner."""
    _read: set[str] = field(default_factory=set, repr=False, compare=False)

    def section(self, name: str, kind: type[T]) -> T:
        """The section ``name``, validated as ``kind``; an error when the entry has none or it does not fit."""
        if name not in self.sections:
            raise ValueError(f"step {self.id!r} has no {name!r} section; it has {sorted(self.sections) or 'none'}")
        try:
            value = TypeAdapter(kind).validate_python(self.sections[name])
        except ValidationError as exc:
            raise ValueError(f"step {self.id!r}, section {name!r}: {exc}") from None
        self._read.add(name)
        return value

    def unread(self) -> list[str]:
        """Its sections nothing has read so far."""
        return [name for name in self.sections if name not in self._read]


class StepDefinitions(Mapping[str, StepDefinition]):
    """A procedure's step definitions by id, in file order, and its ``lines`` (see the module docstring)."""

    def __init__(self, steps: Mapping[str, StepDefinition], lines: Mapping[str, str]) -> None:
        self._steps = dict(steps)
        self.lines = dict(lines)

    @classmethod
    def load(cls, path: Path | str) -> StepDefinitions:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        lines = raw.pop("lines", None) or {}
        steps = {}
        for step_id, entry in raw.items():
            entry = dict(entry or {})
            shared = {key: entry.pop(key, None) for key in SHARED}
            steps[step_id] = StepDefinition(step_id, **shared, sections=entry)
        return cls(steps, lines)

    def __getitem__(self, step_id: str) -> StepDefinition:
        try:
            return self._steps[step_id]
        except KeyError:
            raise KeyError(f"no definition of step {step_id!r}; the file has {sorted(self._steps)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def plan(self) -> list[tuple[str, str]]:
        """``(step id, what the person does)`` in file order, for :class:`~..runner.SopRunner`'s ``plan``."""
        return [(d.id, d.does) for d in self._steps.values() if d.does]

    def line(self, name: str, **values: object) -> str:
        """One of ``lines``, with its ``{placeholders}`` filled."""
        return self.lines[name].format(**values)

    def unread(self) -> list[str]:
        """The sections nothing has read so far, as ``step.section``."""
        return [f"{d.id}.{name}" for d in self._steps.values() for name in d.unread()]

    def check_read(self) -> None:
        """Raise when a section was never read: call it once the app has built its steps."""
        if unread := self.unread():
            raise ValueError(f"sections nothing reads (misspelled, or their reader not wired): {unread}")
