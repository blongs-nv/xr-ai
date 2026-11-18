# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A read step: read something from the camera into a schema, then derive a result from it
(:func:`~.read_and_derive.read_and_derive`), for example a tea's label and its brewing settings. How an evaluation case
of a read step is scored is in :mod:`.read_scoring`.

Its definition in the procedure's YAML: the ``read`` section (:class:`ReadSection`: ``question`` for the perceiving
model, ``instructions`` for deriving the result), ``start_line``, and ``done_line``. ``values`` fill the
``{placeholders}`` in the lines; ``done_line`` also gets ``reading`` (what was read) and ``derived`` (what was
derived)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from xr_ai_app import Role

from ...steps.step_api import StepEnvironment, fill_placeholders
from ...steps.step_definitions import StepDefinition
from .read_and_derive import ReadResult, read_and_derive


class ReadSection(BaseModel):
    """A read step's section of its definition (``read:``)."""

    model_config = ConfigDict(extra="forbid")
    question: str
    """What the perceiving model reads from each frame into the reading schema."""
    instructions: str
    """How the reasoning model derives the derived schema from what was read."""


@dataclass(frozen=True, slots=True)
class ReadModels:
    """The models a read step uses, by what each does."""

    perceive: Role
    """Reads each frame into the reading schema (a vision model)."""
    reason: Role
    """Derives the result from the reading (a reasoning text model)."""


class ReadSettings(BaseModel):
    """How a read step looks (an app's settings may embed this)."""

    model_config = ConfigDict(extra="forbid")
    interval_s: float = 2.0
    """How often a frame is read."""
    timeout_s: float = 30.0
    """Bound on one model call."""


class ReadStep:
    """Read ``reading_schema`` from the camera until ``is_enough``; then derive ``derived_schema`` from it.

    Built from the step's ``definition`` and the ``models`` it uses. ``is_enough(reading)`` decides in code whether a
    reading is enough to go on; ``retrieve(reading)``, when given, returns text passages to derive from (a search
    over a document store). ``values`` fill the placeholders in its lines.
    """

    scoring: ClassVar[str] = f"{__package__}.read_scoring"
    section: ClassVar[str] = "read"

    def __init__(
        self,
        definition: StepDefinition,
        models: ReadModels,
        *,
        reading_schema: type[BaseModel],
        derived_schema: type[BaseModel],
        is_enough: Callable[[Any], bool] = lambda _reading: True,
        retrieve: Callable[[Any], Awaitable[Sequence[str]]] | None = None,
        values: Mapping[str, object] | None = None,
        settings: ReadSettings | None = None,
    ) -> None:
        self.values = dict(values or {})
        words = definition.section(self.section, ReadSection)
        self.id = definition.id
        self.start_line = fill_placeholders(definition.start_line, self.values)
        self.question = words.question
        self.instructions = words.instructions
        self.models = models
        self.reading_schema = reading_schema
        self.derived_schema = derived_schema
        self.is_enough = is_enough
        self.retrieve = retrieve
        self.settings = settings or ReadSettings()
        self._done_line = definition.done_line

    def done_line(self, result: ReadResult[Any, Any]) -> str | None:
        """``done_line``, filled with the values, ``reading`` and ``derived``."""
        return fill_placeholders(self._done_line, {**self.values, "reading": result.reading, "derived": result.derived})

    async def run(self, environment: StepEnvironment) -> ReadResult[Any, Any]:
        async def on_state(state: ReadResult[Any, Any]) -> None:
            await environment.report_progress(self.id, state)

        return await read_and_derive(
            environment.frames,  # type: ignore[arg-type]  # any Frames: the participant's, or a recorded scene
            perceive=self.models.perceive,
            reason=self.models.reason,
            question=self.question,
            reading_schema=self.reading_schema,
            derived_schema=self.derived_schema,
            instructions=self.instructions,
            is_enough=self.is_enough,
            retrieve=self.retrieve,
            interval_s=self.settings.interval_s,
            timeout_s=self.settings.timeout_s,
            on_state=on_state,
        )
