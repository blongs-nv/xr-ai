# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A transform step: a reasoning model turns one schema's instance into another's, following the step's prompt.

No camera. Its input is each run's inputs (an instance of ``input_schema``, from earlier steps); its result is an
instance of ``output_schema``. The two may be the same schema: then, with ``keep_input_values``, the step fills in
what the input leaves empty (settings a package does not print) and keeps, in code, every value the input has, so the
model never overwrites what was read. ``required_fields`` must be set in the result; a reply that leaves one empty is
asked again, up to ``attempts`` times, and then the step fails.

When its prompt allows it, the model may ask the person one short question instead of answering (at most
``max_questions`` per run): the step asks it through its environment, and the question and answer go into the next
prompt. The step is not skippable: it does not wait on the person to do anything, and later steps need its result.

Its definition in the procedure's YAML: the ``transform`` section (:class:`TransformSection`: ``instructions``, the
prompt), ``start_line``, and ``done_line``, whose placeholders name fields or properties of its inputs and ``output``
(the result). How an evaluation case of it is scored is in :mod:`.transform_scoring`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, create_model
from xr_ai_app import Role, StructuredOutputError, clock

from ...steps.step_api import StepEnvironment, check_inputs, check_placeholders, fill_placeholders
from ...steps.step_definitions import StepDefinition


class TransformSection(BaseModel):
    """A transform step's section of its definition (``transform:``)."""

    model_config = ConfigDict(extra="forbid")
    instructions: str
    """The prompt: what to produce from the input, and when the person may be asked."""


class TransformSettings(BaseModel):
    """How a transform step works (an app's settings may embed this)."""

    model_config = ConfigDict(extra="forbid")
    max_questions: int = 1
    """How many questions it may ask the person per run (0: never)."""
    attempts: int = 3
    """Replies it asks for before failing (an unreadable reply, or one missing a required field)."""
    timeout_s: float = 60.0
    """Bound on one model call."""
    thinking: bool | None = None
    """Whether the model reasons before replying (``None``: the role's default)."""


@dataclass(frozen=True, slots=True)
class TransformState:
    """The state of one transform, reported as it goes and when it completes."""

    status: Literal["working", "asking", "completed"]
    output: BaseModel | None
    filled: tuple[str, ...]
    """The output's fields the model set (with ``keep_input_values``: the ones the input left empty)."""
    asked: tuple[tuple[str, str | None], ...]
    """The questions asked and their answers (``None``: none came)."""
    started_s: float
    updated_s: float

    def summary(self) -> str:
        if self.status == "completed" and self.output is not None:
            filled = f" (filled in: {', '.join(self.filled)})" if self.filled else ""
            return f"completed: {self.output.model_dump_json(exclude_none=True)}{filled}."
        if self.status == "asking":
            return f"asking the person: {self.asked[-1][0]}"
        return "working it out."


class TransformStep:
    """Turn each run's input into an ``output_schema`` instance with a reasoning model; see the module docstring.

    Built from the step's ``definition`` and ``reason``, the reasoning text model that does it.
    """

    scoring: ClassVar[str] = f"{__package__}.transform_scoring"
    section: ClassVar[str] = "transform"
    skippable: ClassVar[bool] = False

    def __init__(
        self,
        definition: StepDefinition,
        reason: Role,
        *,
        input_schema: type[BaseModel],
        output_schema: type[BaseModel],
        keep_input_values: bool = False,
        required_fields: tuple[str, ...] = (),
        settings: TransformSettings | None = None,
    ) -> None:
        words = definition.section(self.section, TransformSection)
        unknown = [name for name in required_fields if name not in output_schema.model_fields]
        if unknown:
            raise ValueError(f"step {definition.id!r}: required_fields {unknown} are not fields of the output schema")
        self.id = definition.id
        self.inputs_schema = input_schema
        self.output_schema = output_schema
        self.instructions = words.instructions
        self.reason = reason
        self.keep_input_values = keep_input_values
        self.required_fields = required_fields
        self.settings = settings or TransformSettings()
        self._start_line = definition.start_line
        self._done_line = definition.done_line
        check_placeholders(self.id, "instructions", words.instructions, input_schema)
        check_placeholders(self.id, "start_line", definition.start_line, input_schema)
        check_placeholders(self.id, "done_line", definition.done_line, input_schema, also=("output",))
        self._reply = create_model(
            f"{output_schema.__name__}Reply",
            output=(output_schema | None, None),
            question=(str | None, None),
        )
        self._reply_without_question = create_model(f"{output_schema.__name__}Reply", output=(output_schema, ...))

    def start_line(self, inputs: BaseModel | None = None) -> str | None:
        return fill_placeholders(self._start_line, inputs)

    def done_line(self, inputs: BaseModel | None, result: BaseModel) -> str | None:
        """``done_line``, filled with the inputs and ``output``."""
        return fill_placeholders(self._done_line, inputs, output=result)

    def kept(self, inputs: BaseModel) -> dict[str, Any]:
        """The input's values the output keeps: its non-empty fields the output schema shares (with
        ``keep_input_values``; otherwise none)."""
        if not self.keep_input_values:
            return {}
        shared = [name for name in self.output_schema.model_fields if name in type(inputs).model_fields]
        return {name: value for name in shared if (value := getattr(inputs, name)) is not None}

    def prompt(
        self, inputs: BaseModel, asked: list[tuple[str, str | None]], *, may_ask: bool, missing: list[str]
    ) -> str:
        """What the model is asked: the instructions, the input, what the person said, and what it must do."""
        parts = [
            fill_placeholders(self.instructions, inputs) or "",
            f"The input ({type(inputs).__name__}):\n{inputs.model_dump_json(exclude_none=True)}",
        ]
        kept = self.kept(inputs)
        if kept:
            parts.append(f"These fields are already known and are kept as they are: {', '.join(kept)}.")
        if self.required_fields:
            parts.append(f"The output must set: {', '.join(self.required_fields)}.")
        for question, answer in asked:
            heard = f'They answered: "{answer}"' if answer is not None else "No answer came."
            parts.append(f'You asked the person: "{question}" {heard}')
        if missing:
            parts.append(f"The last reply left these empty: {', '.join(missing)}. Set them.")
        if may_ask:
            parts.append(
                "If, and only if, the instructions say the person should be asked and you cannot give the output "
                "without them, set question to one short spoken question and leave output empty; otherwise set "
                "output."
            )
        return "\n\n".join(parts)

    async def run(self, environment: StepEnvironment, inputs: BaseModel | None = None) -> BaseModel:
        """Ask the model for the output (and, when it asks, the person); return it with the input's kept values."""
        check_inputs(self.id, self.inputs_schema, inputs)
        assert inputs is not None  # noqa: S101 - check_inputs: a transform always has an input schema
        settings = self.settings
        started = clock.now()
        asked: list[tuple[str, str | None]] = []
        missing: list[str] = []
        attempts = 0
        while attempts < settings.attempts:
            may_ask = len(asked) < settings.max_questions
            text = self.prompt(inputs, asked, may_ask=may_ask, missing=missing)
            schema = self._reply if may_ask else self._reply_without_question
            try:
                reply = await self.reason.ask(
                    None, text, response_format=schema, thinking=settings.thinking, timeout_s=settings.timeout_s
                )
            except (StructuredOutputError, TimeoutError):
                attempts += 1
                continue
            question = getattr(reply, "question", None)
            if reply.output is None and may_ask and question:
                asked.append((question, None))
                await environment.report_progress(
                    self.id, TransformState("asking", None, (), tuple(asked), started, clock.now())
                )
                asked[-1] = (question, await environment.ask(question))
                continue  # a question is not an attempt
            attempts += 1
            if reply.output is None:
                continue
            output = reply.output.model_copy(update=self.kept(inputs))
            missing = [name for name in self.required_fields if getattr(output, name) is None]
            if missing:
                continue
            kept = self.kept(inputs)
            filled = tuple(
                n for n in self.output_schema.model_fields if n not in kept and getattr(output, n) is not None
            )
            await environment.report_progress(
                self.id, TransformState("completed", output, filled, tuple(asked), started, clock.now())
            )
            return output
        raise RuntimeError(
            f"step {self.id!r}: no complete {self.output_schema.__name__} after {settings.attempts} replies"
        )
