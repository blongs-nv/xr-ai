# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera frames: ``Frame`` and each participant's ``FrameStream`` (``p.frames``)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from . import _tracing, _virtual


@dataclass(frozen=True, slots=True)
class Frame:
    """One camera frame owned by its participant's scope."""

    image: bytes
    """Encoded image bytes."""

    timestamp_us: int
    """Capture time as Unix microseconds."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    mime_type: str = "image/jpeg"
    """Media type of ``image``."""

    sequence: int = 0
    """Monotonic per-participant frame counter; used to skip duplicates and count distinct frames."""


STALE_FRAME_S: Final = 2.0
"""Seconds after which ``FrameStream.every`` treats a frame as stale (or twice its interval, if longer)."""


class FrameStream:
    """The frames of one participant (``p.frames``).

    Two ways to read it: :meth:`current` (one frame now, for a question) and :meth:`every` (a frame per
    tick, for a monitor). The app pushes each hub frame into the stream; only the newest frame is kept,
    so a slow reader never builds a backlog. The stream is closed when the participant leaves:
    :meth:`every` then ends.

    A device that does not stream can still be asked for one picture: :meth:`current` and each tick
    of :meth:`every` ask for it when no fresh frame exists (the web client's on-demand camera mode).
    """

    def __init__(self) -> None:
        self._capture: Callable[[], Awaitable[Frame | None]] | None = None
        """Set by the device side (``Hub``, ``FakeHub``): take one picture now, ``None`` when the device cannot."""
        self._latest: Frame | None = None
        self._version = 0
        self._returned = 0
        """Version of the newest frame handed out (``_fresh``, ``_next``)."""
        self._changed = asyncio.Event()
        self._closed = False

    # -- app side (private) --------------------------------------------------------------

    def _push(self, frame: Frame) -> None:
        """Make ``frame`` the newest frame and wake readers."""
        if self._closed:
            return
        self._latest = frame
        self._version += 1
        self._wake()

    def _clear(self) -> None:
        """Drop the held frame but keep the stream open (the last connection ended; the participant stays enrolled)."""
        self._latest = None
        self._wake()

    def _close(self) -> None:
        """Release the held frame and end every reader (leave step 5)."""
        self._closed = True
        self._latest = None
        self._wake()

    def _wake(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    async def _newer_than(self, version: int, timeout_s: float | None) -> bool:
        """Wait until a frame newer than ``version`` exists; ``False`` on timeout or close."""
        deadline = None if timeout_s is None else asyncio.get_running_loop().time() + timeout_s
        while not self._closed and self._version <= version:
            changed = self._changed
            remaining = None if deadline is None else deadline - asyncio.get_running_loop().time()
            if remaining is not None and remaining <= 0:
                return False
            try:
                await asyncio.wait_for(changed.wait(), remaining)
            except TimeoutError:
                return False
        return not self._closed

    async def _fresh(self, *, max_age_s: float | None = None) -> Frame | None:
        """The newest streamed frame, or ``None`` when none exists or it is older than ``max_age_s``."""
        frame = self._latest
        if frame is None:
            return None
        if max_age_s is not None and _virtual.wall_us() - frame.timestamp_us > max_age_s * 1_000_000:
            return None
        self._returned = self._version
        return frame

    async def _next(self, *, timeout_s: float = 5.0) -> Frame | None:
        """Wait for a streamed frame newer than any returned so far; ``None`` on timeout."""
        if not await self._newer_than(self._returned, timeout_s):
            return None
        self._returned = self._version
        return self._latest

    async def _picture(self, timeout_s: float) -> Frame | None:
        """Ask the device for one picture (traced); ``None`` when it cannot give one within ``timeout_s``."""
        if self._capture is None:
            return None
        try:
            frame = await asyncio.wait_for(self._capture(), timeout_s)
        except TimeoutError:
            frame = None
        _tracing.emit("frame", "capture", ok=frame is not None)
        return frame

    # -- app code ------------------------------------------------------------------------

    async def current(self, *, max_age_s: float = 2.0, timeout_s: float = 5.0) -> Frame | None:
        """The newest frame if at most ``max_age_s`` old, else the next one within ``timeout_s``; ``None`` when none
        arrives.

        As upstream's ``CurrentFrameTool``: with no fresh frame the device is first asked for one picture (a client in
        on-demand camera mode turns its camera on for one frame), and only when it cannot give one (a live-streaming
        client, no client) does the stream answer. The picture is not pushed into the stream.
        """
        frame = await self._fresh(max_age_s=max_age_s)
        if frame is not None:
            return frame
        frame = await self._picture(timeout_s)
        if frame is not None:
            return frame
        return await self._next(timeout_s=timeout_s)

    async def every(self, interval_s: float, *, timeout_s: float = 5.0) -> AsyncIterator[Frame]:
        """Yield a recent frame at most every ``interval_s`` seconds until the participant leaves.

        With a live stream: distinct streamed frames. The first is yielded as soon as one exists; after each frame
        the stream waits out the interval and then yields the newest frame that arrived, so frames in between are
        skipped. A frame is distinct when it is newer than the last one yielded and, when it carries a ``sequence``
        above zero, has a different ``sequence``. A frame older than ``max(2 * interval_s, STALE_FRAME_S)`` is never
        yielded, so a camera that stopped delivering yields nothing instead of repeating its last frame.

        With no stream (a client in on-demand camera mode): a tick that finds no streamed frame within that age asks
        the device for one picture, as :meth:`current` does, and yields it; a picture that does not come within
        ``timeout_s`` yields nothing for that tick, and the ticks go on. Once a tick was a picture (or a picture that
        timed out), the next tick asks at once when no newer streamed frame is there; after a device said it has no
        picture it waits for the stream again first. Without a device that can take pictures it only waits for the
        stream. Breaking out of the loop releases nothing the caller must clean up.
        """
        if interval_s < 0:
            raise ValueError("interval_s must not be negative")
        loop = asyncio.get_running_loop()
        max_age_s = max(2 * interval_s, STALE_FRAME_S)
        cursor = self._version if self._latest is None else self._version - 1
        last_sequence = 0
        streaming = True  # False once a tick was a picture
        due = loop.time()
        while not self._closed:
            delay = due - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            wait = None if self._capture is None else (max_age_s if streaming else 0.0)
            if not await self._newer_than(cursor, wait):
                if self._closed:
                    return
                asked = loop.time()
                frame = await self._picture(timeout_s)
                due = loop.time() + interval_s
                if frame is not None or due - interval_s - asked >= timeout_s:
                    streaming = False  # on demand (a picture, or one too slow): the next tick asks at once
                if frame is not None:
                    yield frame
                continue
            frame = self._latest
            cursor = self._version
            if frame is None:
                return
            if _virtual.wall_us() - frame.timestamp_us > max_age_s * 1_000_000:
                continue  # stale: the camera is not delivering
            if frame.sequence > 0 and frame.sequence == last_sequence:
                continue  # the same frame pushed again
            last_sequence, streaming = frame.sequence, True
            due = loop.time() + interval_s
            yield frame


__all__ = ["STALE_FRAME_S", "Frame", "FrameStream"]
