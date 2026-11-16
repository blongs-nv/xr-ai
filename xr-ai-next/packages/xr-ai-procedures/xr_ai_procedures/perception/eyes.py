# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera pacing for a look: fresh, distinct frames at a steady rate, and the watching policy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from xr_ai_next import clock

if TYPE_CHECKING:
    from xr_ai_next import Frame, FrameStream

__all__ = ["Eyes", "FrameSource", "Watching"]

_PICTURE_TIMEOUT_S = 5.0
"""How long a look waits for one picture by default: a client in on-demand mode may turn its camera on for it."""


class FrameSource(Protocol):
    """What a look reads frames from: :class:`Eyes`, or anything with ``async next() -> frame | None``."""

    async def next(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class Watching:
    """How a camera look watches; one value is usually shared by every look of an app.

    A look is confirmed when ``required_positives`` of the last ``window`` usable looks say yes. With no usable look for
    ``stuck_after_s`` it gives up (``Stuck``). A judge lacking the same fact ``missing_after`` looks in a row makes it
    ask for that fact (``Needs``). It looks at most once per ``interval_s`` (and waits that long for a new frame), at
    frames at most ``max_age_s`` old.
    """

    required_positives: int = 2
    window: int = 3
    stuck_after_s: float = 20.0
    missing_after: int = 2
    interval_s: float = 1.0
    max_age_s: float = 3.0

    def __post_init__(self) -> None:
        if not 1 <= self.required_positives <= self.window:
            raise ValueError("required_positives must be at least 1 and fit in the window")
        if self.interval_s <= 0 or self.stuck_after_s <= 0 or self.max_age_s <= 0:
            raise ValueError("interval_s, stuck_after_s and max_age_s must be positive")
        if self.missing_after < 1:
            raise ValueError("missing_after must be at least 1")


class Eyes:
    """Camera frames for one look: the newest fresh frame, never the same image twice, at most one look per
    ``interval_s``, the first look at once. With no streamed frame it asks the device for one picture
    (``FrameStream.current``), so a look also works in on-demand camera mode.

    ``next()`` returns ``None`` when no frame came within ``wait_s`` (default ``max(interval_s, 5)``), or only the one
    the last look saw (a stream slower than one frame per interval); the look counts as unseen.
    """

    def __init__(
        self,
        frames: FrameStream,
        *,
        interval_s: float = 1.0,
        max_age_s: float = 3.0,
        wait_s: float | None = None,
        now: Callable[[], float] = clock.now,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._frames = frames
        self._interval = interval_s
        self._max_age = max_age_s
        self._wait = max(interval_s, _PICTURE_TIMEOUT_S) if wait_s is None else wait_s
        self._clock = now
        self._last: float | None = None
        self._seen: tuple[int, int] | None = None

    @classmethod
    def watching(cls, frames: FrameStream, watching: Watching) -> Eyes:
        """Eyes paced by a :class:`Watching` policy; a look waits for a frame at most until the policy would give up."""
        wait_s = max(watching.interval_s, min(_PICTURE_TIMEOUT_S, watching.stuck_after_s))
        return cls(frames, interval_s=watching.interval_s, max_age_s=watching.max_age_s, wait_s=wait_s)

    async def next(self) -> Frame | None:
        """A fresh frame this look has not seen yet, or ``None``."""
        if self._last is not None:
            wait = self._interval - (self._clock() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
        self._last = self._clock()
        # current(): the newest fresh frame, or, with none (a client in on-demand camera mode streams nothing), one
        # picture asked of the device (a browser turns its camera on for it), or the next streamed frame.
        frame = await self._frames.current(max_age_s=self._max_age, timeout_s=self._wait)
        if frame is None or _key(frame) == self._seen or _age_s(frame) > self._max_age:
            return None
        self._seen = _key(frame)
        return frame


def _key(frame: Frame) -> tuple[int, int]:
    return frame.sequence, frame.timestamp_us


def _age_s(frame: Frame) -> float:
    return (clock.wall_us() - frame.timestamp_us) / 1_000_000
