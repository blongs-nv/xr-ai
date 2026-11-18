# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A read step: read something from the camera into a schema until what has been read is enough
(:func:`~.read_until_enough.read_until_enough`), for example what a tea's package says. Its result is the reading, an
instance of the schema. How an evaluation case of a read step is scored is in :mod:`.read_scoring`.

Its words (:class:`ReadWords`): ``question``, what the perceiving model reads from each frame, and what every step
says (``does``, ``start_line``, ``done_line``, whose placeholders name fields or properties of its inputs and
``reading``, what was read)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from xr_ai_app import Role

from ...steps.step_api import StepEnvironment, check_inputs, check_placeholders, fill_placeholders
from ...steps.step_words import StepWords
from .read_until_enough import ReadState, read_until_enough


class ReadWords(StepWords):
    """A read step's words: what every step says, and its ``question``."""

    question: str
    """What the perceiving model reads from each frame into the reading schema."""


class ReadSettings(BaseModel):
    """How a read step looks (an app's settings may embed this)."""

    model_config = ConfigDict(extra="forbid")
    interval_s: float = 0.0
    """At least this long between two frames read (0: the next new frame as soon as the last one is read)."""
    timeout_s: float = 30.0
    """Bound on one model call."""


class ReadStep:
    """Read from the camera until ``is_enough(reading)``; the result is the reading (see this package's README for
    how to shape the schema).

    ``perceive`` reads each frame (a vision model). Reading a frame into a schema means judging what each field may
    take (what is printed on the package, not a display behind it): either a model that reasons, or a schema that
    makes it decide what is in view and copy what it sees before it interprets it, without reasoning (the tea
    sample's). ``reply_schema`` is what the model fills in for one frame, ``extract`` turns a reply into the record
    that is kept (default: the reply itself; fields read in different frames are merged), and ``is_enough`` decides in
    code whether what has been read is enough to go on. ``inputs_schema`` is what each run starts with (``None``:
    nothing).
    """

    scoring: ClassVar[str] = f"{__package__}.read_scoring"

    def __init__(
        self,
        step_id: str,
        words: ReadWords,
        perceive: Role,
        *,
        reply_schema: type[BaseModel],
        extract: Callable[[Any], BaseModel] | None = None,
        is_enough: Callable[[Any], bool] = lambda _reading: True,
        inputs_schema: type[BaseModel] | None = None,
        settings: ReadSettings | None = None,
    ) -> None:
        self.id = step_id
        self.words = words
        self.inputs_schema = inputs_schema
        self.question = words.question
        self.perceive = perceive
        self.reply_schema = reply_schema
        self.extract = extract
        self.is_enough = is_enough
        self.settings = settings or ReadSettings()
        check_placeholders(self.id, "question", words.question, inputs_schema)
        check_placeholders(self.id, "start_line", words.start_line, inputs_schema)
        check_placeholders(self.id, "done_line", words.done_line, inputs_schema, also=("reading",))

    def start_line(self, inputs: BaseModel | None = None) -> str | None:
        return fill_placeholders(self.words.start_line, inputs)

    def done_line(self, inputs: BaseModel | None, result: BaseModel) -> str | None:
        """``done_line``, filled with the inputs and ``reading``."""
        return fill_placeholders(self.words.done_line, inputs, reading=result)

    async def run(self, environment: StepEnvironment, inputs: BaseModel | None = None) -> Any:
        check_inputs(self.id, self.inputs_schema, inputs)

        async def on_state(state: ReadState[Any]) -> None:
            await environment.report_progress(self.id, state)

        return await read_until_enough(
            environment.frames,  # type: ignore[arg-type]  # any Frames: the participant's, or a recorded scene
            perceive=self.perceive,
            question=fill_placeholders(self.question, inputs) or "",
            reply_schema=self.reply_schema,
            extract=self.extract,
            is_enough=self.is_enough,
            interval_s=self.settings.interval_s,
            timeout_s=self.settings.timeout_s,
            on_state=on_state,
        )
