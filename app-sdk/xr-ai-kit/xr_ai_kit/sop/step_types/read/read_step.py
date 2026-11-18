# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A read step: read something from the camera into a schema until what has been read is enough
(:func:`~.read_until_enough.read_until_enough`), for example what a tea's package says. Its result is the reading, an
instance of the schema. How an evaluation case of a read step is scored is in :mod:`.read_scoring`.

Its definition in the procedure's YAML: the ``read`` section (:class:`ReadSection`: ``question``, what the perceiving
model reads from each frame), ``start_line``, and ``done_line``, whose placeholders name fields or properties of its
inputs and ``reading`` (what was read)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from xr_ai_app import Role

from ...steps.step_api import StepEnvironment, check_inputs, check_placeholders, fill_placeholders
from ...steps.step_definitions import StepDefinition
from .read_until_enough import ReadState, read_until_enough


class ReadSection(BaseModel):
    """A read step's section of its definition (``read:``)."""

    model_config = ConfigDict(extra="forbid")
    question: str
    """What the perceiving model reads from each frame into the reading schema."""


class ReadSettings(BaseModel):
    """How a read step looks (an app's settings may embed this)."""

    model_config = ConfigDict(extra="forbid")
    interval_s: float = 2.0
    """How often a frame is read."""
    timeout_s: float = 30.0
    """Bound on one model call."""


class ReadStep:
    """Read ``reading_schema`` from the camera until ``is_enough(reading)``; the result is the reading.

    ``perceive`` reads each frame (a vision model). ``is_enough`` decides in code whether what has been read is enough
    to go on (fields read in different frames are merged). ``inputs_schema`` is what each run starts with (``None``:
    nothing).
    """

    scoring: ClassVar[str] = f"{__package__}.read_scoring"
    section: ClassVar[str] = "read"

    def __init__(
        self,
        definition: StepDefinition,
        perceive: Role,
        *,
        reading_schema: type[BaseModel],
        is_enough: Callable[[Any], bool] = lambda _reading: True,
        inputs_schema: type[BaseModel] | None = None,
        settings: ReadSettings | None = None,
    ) -> None:
        words = definition.section(self.section, ReadSection)
        self.id = definition.id
        self.inputs_schema = inputs_schema
        self.question = words.question
        self.perceive = perceive
        self.reading_schema = reading_schema
        self.is_enough = is_enough
        self.settings = settings or ReadSettings()
        self._start_line = definition.start_line
        self._done_line = definition.done_line
        check_placeholders(self.id, "question", words.question, inputs_schema)
        check_placeholders(self.id, "start_line", definition.start_line, inputs_schema)
        check_placeholders(self.id, "done_line", definition.done_line, inputs_schema, also=("reading",))

    def start_line(self, inputs: BaseModel | None = None) -> str | None:
        return fill_placeholders(self._start_line, inputs)

    def done_line(self, inputs: BaseModel | None, result: BaseModel) -> str | None:
        """``done_line``, filled with the inputs and ``reading``."""
        return fill_placeholders(self._done_line, inputs, reading=result)

    async def run(self, environment: StepEnvironment, inputs: BaseModel | None = None) -> Any:
        check_inputs(self.id, self.inputs_schema, inputs)

        async def on_state(state: ReadState[Any]) -> None:
            await environment.report_progress(self.id, state)

        return await read_until_enough(
            environment.frames,  # type: ignore[arg-type]  # any Frames: the participant's, or a recorded scene
            perceive=self.perceive,
            question=fill_placeholders(self.question, inputs) or "",
            reading_schema=self.reading_schema,
            is_enough=self.is_enough,
            interval_s=self.settings.interval_s,
            timeout_s=self.settings.timeout_s,
            on_state=on_state,
        )
