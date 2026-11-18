# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read something from the camera into a schema, frame after frame, until what has been read is enough.

The schema is the app's (``reading_schema``, for example what a tea's package says). Its fields should default to
``None``: a frame shows only part of the thing (the front of a box, then its back), so each field keeps the last
value read for it, and a later frame that does not show a field does not erase it. Whether what has been read is
enough to go on is the app's call, in code (``is_enough``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel
from xr_ai_app import FrameStream, Role, StructuredOutputError, clock

R = TypeVar("R", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ReadState(Generic[R]):
    """The state of one read, reported after every frame."""

    status: Literal["reading", "completed"]
    reading: R | None
    """Everything read so far, merged across frames (``None`` until a frame fit the schema)."""

    frames: int
    """Frames read so far."""

    started_s: float
    updated_s: float

    def summary(self) -> str:
        read = "nothing readable yet" if self.reading is None else _json(self.reading)
        if self.status == "completed":
            return f"completed: read {read}."
        return f"reading ({self.frames} frames): {read}; not enough to go on yet."


def _json(model: BaseModel) -> str:
    return model.model_dump_json(exclude_none=True)


def merge(earlier: R | None, latest: R) -> R:
    """``latest``, keeping each field of ``earlier`` that ``latest`` leaves empty."""
    if earlier is None:
        return latest
    seen = {name: value for name in type(latest).model_fields if (value := getattr(latest, name)) is not None}
    return earlier.model_copy(update=seen)


async def read_until_enough(
    frames: FrameStream,
    *,
    perceive: Role,
    question: str,
    reading_schema: type[R],
    is_enough: Callable[[R], bool] = lambda _reading: True,
    interval_s: float = 2.0,
    timeout_s: float | None = 15.0,
    on_state: Callable[[ReadState[R]], Awaitable[None]] | None = None,
) -> R:
    """Read a frame every ``interval_s`` into ``reading_schema`` with ``perceive`` (asked ``question``), merging what
    each frame shows (:func:`merge`), until ``is_enough`` says it is; return what was read. ``on_state`` is awaited in
    this task after every frame."""
    started = clock.now()
    count = 0
    reading: R | None = None
    async for frame in frames.every(interval_s, timeout_s=timeout_s or 5.0):
        count += 1
        try:
            latest = await perceive.ask(frame, question, response_format=reading_schema, timeout_s=timeout_s)
        except StructuredOutputError:
            latest = None
        if latest is not None:
            reading = merge(reading, latest)
        done = reading is not None and is_enough(reading)
        if on_state is not None:
            await on_state(ReadState("completed" if done else "reading", reading, count, started, clock.now()))
        if done:
            assert reading is not None  # noqa: S101 - narrowed for the type checker
            return reading
    raise RuntimeError("the frame stream ended before enough was read")  # the participant left
