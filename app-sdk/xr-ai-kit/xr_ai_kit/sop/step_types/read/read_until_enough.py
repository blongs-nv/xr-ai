# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read something from the camera into a schema, frame after frame, until what has been read is enough.

The schema is the app's: ``reply_schema``, what the model fills in for one frame (it may be shaped to guide the
model: a gate such as "is the package in view?", the words copied before they are interpreted; see this package's
README), and ``extract``, which turns one reply into the record that is kept (default: the reply itself). The
record's fields should default to ``None``: a frame shows only part of the thing (the front of a box, then its
back), so each field keeps the last value read for it, and a later frame that does not show a field does not erase
it. Whether what has been read is enough to go on is the app's call, in code (``is_enough``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel
from xr_ai_app import FrameStream, Role, StructuredOutputError, clock

R = TypeVar("R", bound=BaseModel)
S = TypeVar("S", bound=BaseModel)


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
    reply_schema: type[S],
    extract: Callable[[S], R] | None = None,
    is_enough: Callable[[R], bool] = lambda _reading: True,
    interval_s: float = 0.0,
    timeout_s: float | None = 15.0,
    on_state: Callable[[ReadState[R]], Awaitable[None]] | None = None,
) -> R:
    """Read frame after frame with ``perceive`` (asked ``question``, replying in ``reply_schema``), turn each reply
    into a record (``extract``), and merge what each frame shows (:func:`merge`) until ``is_enough`` says it is;
    return what was read. Each read takes the newest frame not
    read yet, at least ``interval_s`` after the last (0: as soon as the last read is done). ``on_state`` is awaited in
    this task after every frame."""
    started = clock.now()
    count = 0
    reading: R | None = None
    async for frame in frames.every(interval_s, timeout_s=timeout_s or 5.0):
        count += 1
        try:
            reply = await perceive.ask(frame, question, response_format=reply_schema, timeout_s=timeout_s)
        except StructuredOutputError:
            reply = None
        if reply is not None:
            latest = extract(reply) if extract is not None else reply
            reading = merge(reading, latest)  # type: ignore[arg-type]  # without extract, the reply is the record
        done = reading is not None and is_enough(reading)
        if on_state is not None:
            await on_state(ReadState("completed" if done else "reading", reading, count, started, clock.now()))
        if done:
            assert reading is not None  # noqa: S101 - narrowed for the type checker
            return reading
    raise RuntimeError("the frame stream ended before enough was read")  # the participant left
