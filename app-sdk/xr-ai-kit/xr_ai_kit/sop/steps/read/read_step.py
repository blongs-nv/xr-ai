# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A read step: read something from the camera into a schema, then derive a result from it (:func:`~.identify.
identify`), for example a tea's label and its brewing settings. How an evaluation case of a read step is scored is in
:mod:`.read_scoring`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..step_api import Roles, StepIO
from .identify import Identification, identify


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
