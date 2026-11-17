# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A procedure's steps written in one YAML file (:class:`StepDefinitions`), and the step objects built from them.

Only words live in the file; which type each step is, its schemas, code hooks and escalator are chosen where the step
is built (``definitions.watch("heat_water", roles, needs=...).escalating(definitions.escalator("heat_water", r))``).
One entry per step, keyed by its id, and ``lines`` for whatever else the procedure says::

    fill_water:
      does: fill the kettle with fresh water          # the plan the conversation reads
      say: Fill a kettle with fresh water and keep it in view.
      question: Describe whether water is visibly inside the kettle ...   # a watch step's check
      criteria: {instructions: ..., positive: ..., negative: ..., unknown: ...}
      escalation: The kettle has to be in the camera's view ...           # optional: its escalator's guidance
    identify:
      does: show the tea's label
      say: Show the tea's label clearly.
      question: Read only clearly legible text ...    # a read step
      instructions: Recommend brewing settings ...
    steep:
      does: wait while the tea steeps
      say: I'll time {minutes:g} minutes from now.    # a wait step: words only
    lines:
      water_seen: Water is visible in the vessel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict
from xr_ai_app import Role

from ..read import ReadStep
from ..wait import WaitStep
from ..watch import Check, Criteria, EscalationSettings, Escalator, WatchSettings, WatchStep
from .step_api import Roles


class _Criteria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str
    positive: str
    negative: str
    unknown: str


class StepDefinition(BaseModel):
    """One step's definition: its words. Which fields it has depends on the step's type."""

    model_config = ConfigDict(extra="forbid")
    does: str | None = None
    """What the person does in this step, for the plan the conversation reads."""
    say: str | None = None
    """The instruction spoken as the step starts."""
    question: str | None = None
    criteria: _Criteria | None = None
    escalation: str | None = None
    instructions: str | None = None


class StepDefinitions:
    """A procedure's step definitions, loaded from YAML (see the module docstring)."""

    def __init__(self, steps: dict[str, StepDefinition], lines: dict[str, str]) -> None:
        self.steps = steps
        self.lines = lines

    @classmethod
    def load(cls, path: Path | str) -> StepDefinitions:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        lines = raw.pop("lines", None) or {}
        return cls({key: StepDefinition.model_validate(value) for key, value in raw.items()}, dict(lines))

    def plan(self) -> list[tuple[str, str]]:
        """``(step id, what the person does)`` in file order, for :class:`~..runner.SopRunner`'s ``plan``."""
        return [(key, words.does) for key, words in self.steps.items() if words.does]

    def line(self, name: str, **values: object) -> str:
        """One of ``lines``, with its ``{placeholders}`` filled."""
        return self.lines[name].format(**values)

    def check(self, step_id: str, **code: Any) -> Check:
        """The step's :class:`~.watch.Check`: its question and criteria, plus what code declares (``needs``,
        ``observation_model``, ``can_assess``, ``reconcile``)."""
        words = self._words(step_id)
        if words.question is None or words.criteria is None:
            raise ValueError(f"{step_id}: a watch step needs question and criteria")
        c = words.criteria
        return Check(words.question, Criteria(c.instructions, c.positive, c.negative, c.unknown), **code)

    def watch(self, step_id: str, roles: Roles, settings: WatchSettings | None = None, **code: Any) -> WatchStep:
        """A :class:`~.watch.WatchStep` from the step's words, without an escalator; ``code`` as for :meth:`check`."""
        return WatchStep(
            step_id,
            self.check(step_id, **code),
            roles,
            say=self._words(step_id).say,
            settings=settings or WatchSettings(),
        )

    def escalator(
        self, step_id: str, role: Role, settings: EscalationSettings | None = None, *, reader: Role | None = None
    ) -> Escalator:
        """An :class:`~.watch.Escalator` for the step, with its ``escalation`` words as guidance; attach it with
        ``WatchStep.escalating``. ``role`` reasons; ``reader`` reads answers (default ``role``)."""
        guidance = self._words(step_id).escalation
        if not guidance:
            raise ValueError(f"{step_id}: an escalator needs the step's escalation words")
        return Escalator(guidance, role, reader, settings or EscalationSettings())

    def read(self, step_id: str, roles: Roles, **code: Any) -> ReadStep:
        """A :class:`~.read.ReadStep` from the step's words; ``code`` gives its schemas and hooks
        (``visual_model``, ``result_model``, ``ready``, ``retrieve``, ``interval_s``, ``timeout_s``)."""
        words = self._words(step_id)
        if words.question is None or words.instructions is None:
            raise ValueError(f"{step_id}: a read step needs question and instructions")
        return ReadStep(step_id, words.question, words.instructions, roles=roles, say=words.say, **code)

    def wait(self, step_id: str, duration_s: float = 0.0, **code: Any) -> WaitStep:
        """A :class:`~.wait.WaitStep` from the step's words (its duration usually set later, ``.lasting(s)``)."""
        return WaitStep(step_id, duration_s, say=self._words(step_id).say, **code)

    def _words(self, step_id: str) -> StepDefinition:
        try:
            return self.steps[step_id]
        except KeyError:
            raise KeyError(f"no words for step {step_id!r}; the file has {sorted(self.steps)}") from None
