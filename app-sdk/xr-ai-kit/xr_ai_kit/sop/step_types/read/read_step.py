# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A read step: read something from the camera into a schema, then derive a result from it (:func:`~.identify.
identify`), for example a tea's label and its brewing settings. How an evaluation case of a read step is scored is in
:mod:`.read_scoring`.

Its definition in the procedure's YAML: the ``read`` section (:class:`ReadWords`: ``question`` for the observing
model, ``instructions`` for deriving the result), and ``done``, filled with the build-time values, ``visual`` (what
was read) and ``result`` (what was derived)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from xr_ai_app import Role

from ...steps.step_api import StepIO
from ...steps.step_definitions import StepDefinition
from .identify import Identification, identify


class ReadWords(BaseModel):
    """A read step's section of its definition (``read:``)."""

    model_config = ConfigDict(extra="forbid")
    question: str
    """What the observing model reads from each frame into ``visual_model``."""
    instructions: str
    """How the reasoning model derives ``result_model`` from what was read."""


@dataclass(frozen=True, slots=True)
class ReadStep:
    """Read ``visual_model`` from the camera until ``ready``; then derive ``result_model``.

    ``observe`` reads each frame (a vision model); ``reason`` derives the result (a reasoning text model).
    """

    scoring: ClassVar[str] = f"{__package__}.read_scoring"
    section: ClassVar[str] = "read"

    id: str
    question: str
    instructions: str
    visual_model: type[BaseModel]
    result_model: type[BaseModel]
    observe: Role
    reason: Role
    say: str | None = None
    done: str | None = None
    values: Mapping[str, object] = field(default_factory=dict)
    ready: Callable[[Any], bool] = lambda _visual: True
    retrieve: Callable[[Any], Awaitable[Sequence[str]]] | None = None
    interval_s: float = 2.0
    timeout_s: float = 30.0

    @classmethod
    def define(
        cls,
        definition: StepDefinition,
        *,
        observe: Role,
        reason: Role,
        visual_model: type[BaseModel],
        result_model: type[BaseModel],
        **code: Any,
    ) -> ReadStep:
        """The step from its definition; ``code`` is the rest (``ready``, ``retrieve``, ``interval_s``,
        ``timeout_s``)."""
        words = definition.section(cls.section, ReadWords)
        return cls(
            definition.id,
            words.question,
            words.instructions,
            visual_model,
            result_model,
            observe,
            reason,
            say=definition.say,
            done=definition.done,
            **code,
        )

    def format(self, **values: object) -> ReadStep:
        """A copy with ``{name}`` placeholders filled in ``say``; ``done`` gets them when it is spoken."""
        say = self.say.format(**values) if self.say else self.say
        return replace(self, say=say, values={**self.values, **values})

    def closing(self, result: Identification[Any, Any]) -> str | None:
        """``done``, filled with the build-time values, ``visual`` and ``result``."""
        if not self.done:
            return None
        return self.done.format(**{**self.values, "visual": result.visual, "result": result.result})

    async def run(self, io: StepIO) -> Identification[Any, Any]:
        return await identify(
            io.frames,  # type: ignore[arg-type]  # any Frames: the participant's, or a recorded scene
            observe=self.observe,
            answer=self.reason,
            question=self.question,
            visual_model=self.visual_model,
            result_model=self.result_model,
            instructions=self.instructions,
            ready=self.ready,
            retrieve=self.retrieve,
            interval_s=self.interval_s,
            timeout_s=self.timeout_s,
            on_state=io.report(self.id),
        )
