# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scripted :class:`~..steps.step_api.StepEnvironment` (and so an escalator's environment): a scene for the camera,
a scripted person, a record of it."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..steps.conversation import FixedConversation
from ..steps.step_api import speaker
from .scene import SceneFrames
from .step_trace import Line, StepTrace


class Person(BaseModel):
    """How the person in a case responds."""

    model_config = ConfigDict(extra="forbid")
    answers: list[str] = []
    """Answers to the step's questions, in order; a question beyond them gets none."""
    answer_after_s: float = 3.0
    """How long the person takes to answer."""
    after_tell: list[Any] | None = None
    """What the camera shows once the person was told something (they did it): scene parts, as ``scene``."""


class ScriptedEnvironment:
    """What a step and its escalator touch in an evaluation (see the module docstring)."""

    def __init__(
        self, step_id: str, scene: SceneFrames, person: Person, *, conversation: str = "", run_summary: str = ""
    ) -> None:
        self.scene = scene
        self.person = person
        self.conversation = FixedConversation(conversation)
        """The case's conversation, fixed for the whole case."""
        self._run_summary = run_summary
        self._answers = list(person.answers)
        self._switched = False
        self.trace = StepTrace(step_id)

    @property
    def frames(self) -> SceneFrames:
        return self.scene

    async def say(self, text: str) -> None:
        self.trace.lines.append(Line(self.scene.now(), "say", text, by=speaker()))
        if self.person.after_tell and not self._switched:
            self._switched = True
            self.scene.switch(self.person.after_tell)

    async def ask(self, question: str) -> str | None:
        asked_at = self.scene.now()
        await asyncio.sleep(self.person.answer_after_s)
        answer = self._answers.pop(0) if self._answers else None
        self.trace.lines.append(Line(asked_at, "ask", question, answer, by=speaker()))
        return answer

    async def report_progress(self, step_id: str, state: Any) -> None:
        self.trace.states.append((self.scene.now(), state))

    async def run_summary(self) -> str:
        """The case's run summary (``""`` when it gives none)."""
        return self._run_summary
