# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scripted :class:`~..step.StepIO`: a scene for the camera, a scripted person, and a record of what happened."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..step import Line, StepTrace
from .scene import SceneFrames


class Person(BaseModel):
    """How the person in a case responds."""

    model_config = ConfigDict(extra="forbid")
    answers: list[str] = []
    """Answers to the step's questions, in order; a question beyond them gets none."""
    answer_after_s: float = 3.0
    """How long the person takes to answer."""
    after_tell: list[Any] | None = None
    """What the camera shows once the person was told something (they did it): scene parts, as ``scene``."""


class ScriptedIO:
    """What a step touches in an evaluation (see the module docstring)."""

    def __init__(
        self, step_id: str, scene: SceneFrames, person: Person, context: Mapping[str, str] | None = None
    ) -> None:
        self.scene = scene
        self.person = person
        self._context = dict(context or {})
        self._answers = list(person.answers)
        self._switched = False
        self.trace = StepTrace(step_id)

    @property
    def frames(self) -> SceneFrames:
        return self.scene

    async def say(self, text: str) -> None:
        self.trace.lines.append(Line(self.scene.now(), "say", text))
        if self.person.after_tell and not self._switched:
            self._switched = True
            self.scene.switch(self.person.after_tell)

    async def ask(self, question: str) -> str | None:
        asked_at = self.scene.now()
        await asyncio.sleep(self.person.answer_after_s)
        answer = self._answers.pop(0) if self._answers else None
        self.trace.lines.append(Line(asked_at, "ask", question, answer))
        return answer

    def report(self, step_id: str | None = None) -> Callable[[Any], Awaitable[None]]:
        async def on_state(state: Any) -> None:
            self.trace.states.append((self.scene.now(), state))

        return on_state

    async def situation(self, sources: Sequence[str]) -> str:
        """The case's text for each source asked for, in order (a source the case does not give is empty)."""
        return "\n\n".join(self._context[s] for s in sources if self._context.get(s))
