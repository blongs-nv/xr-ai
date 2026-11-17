# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A read step: read something from the camera into a schema, then derive a result from it (:func:`~..identify.
identify`), for example a tea's label and its brewing settings. ``Expect`` and :meth:`ReadStep.score` say how an
evaluation case of a read step is scored."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..identify import Identification, identify
from ..step import Roles, StepIO, StepTrace, mentions


class ReadExpect(BaseModel):
    """What an evaluation case may expect of a read step (each key optional)."""

    model_config = ConfigDict(extra="forbid")
    completes: bool | None = None
    reads: dict[str, list[str]] | None = None
    """Per field of what was read: the value mentions one of these words (``[]``: the field is not empty)."""
    derives: dict[str, list[str]] | None = None
    """The same, per field of the derived result."""


@dataclass(frozen=True, slots=True)
class ReadStep:
    """Read ``visual_model`` from the camera until ``ready``; then derive ``result_model`` with the reasoning role."""

    id: str
    question: str
    instructions: str
    visual_model: type[BaseModel]
    result_model: type[BaseModel]
    roles: Roles
    say: str | None = None
    ready: Callable[[Any], bool] = lambda _visual: True
    retrieve: Callable[[Any], Awaitable[Sequence[str]]] | None = None
    interval_s: float = 2.0
    timeout_s: float = 30.0

    Expect = ReadExpect

    async def run(self, io: StepIO) -> Identification[Any, Any]:
        return await identify(
            io.frames,  # type: ignore[arg-type]  # any Frames: the participant's, or a recorded scene
            observe=self.roles.observe,
            answer=self.roles.reason,
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

    @staticmethod
    def score(trace: StepTrace, expect: ReadExpect) -> list[str]:
        """What is wrong with ``trace`` against ``expect`` (empty: the case passed)."""
        problems: list[str] = []
        if expect.completes is not None and trace.completed != expect.completes:
            problems.append("completed" if trace.completed else "did not complete")
        result = trace.result
        for name, wanted, model in (
            ("read", expect.reads, getattr(result, "visual", None)),
            ("derived", expect.derives, getattr(result, "result", None)),
        ):
            for key, words in (wanted or {}).items():
                value = getattr(model, key, None)
                if value is None or not mentions(str(value), words):
                    problems.append(f"{name} {key}={value!r}, expected one of {words or 'any value'}")
        return problems
