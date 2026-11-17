# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A recording, and a scene played from it as a step's camera frames on the loop's clock.

A scene is one part of a recording (``[26, 346]``: played as recorded) or several parts played back to back
(``[[26, 36], [5, 12], [5, 12]]``: in any order, repeats allowed), to build what the recording never had: the person
looking away for a minute while the kettle heats. Scene time starts at the first part's start and runs at the
recording's own pace, so for one part it is the recording's time.

:class:`SceneFrames` serves a scene the way ``Participant.frames`` serves a live camera: :meth:`~SceneFrames.every`
yields the newest frame at once and then after each interval, frames are stamped with ``clock.wall_us()`` (so they
age as live ones do) and numbered (so a repeated part is new frames, not duplicates). When the scene is over it
blocks, like a camera with nothing new; the evaluation ends the step then.
"""

from __future__ import annotations

import asyncio
import csv
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from xr_ai_app import Frame, clock
from xr_ai_app.devkit import image_frame

Segments = list[tuple[float, float]]


def segments(at: Sequence[Any]) -> Segments:
    """``[a, b]`` or ``[[a, b], ...]`` as a list of ``(a, b)``."""
    if not at:
        raise ValueError("a scene needs at least one part")
    parts = [at] if not isinstance(at[0], (list, tuple)) else list(at)
    out = [(float(a), float(b)) for a, b in parts]
    if any(b <= a for a, b in out):
        raise ValueError(f"a scene part must end after it starts: {at}")
    return out


@dataclass(frozen=True, slots=True)
class Shot:
    """One frame of a scene: when it shows (scene time), and where it came from."""

    t_s: float
    source_s: float
    path: Path


class Recording:
    """A recording's frames: an index CSV with ``source_time_s`` and ``frame_image`` (paths relative to the CSV)."""

    def __init__(self, frames: list[tuple[float, Path]]) -> None:
        if not frames:
            raise ValueError("the recording has no frames")
        self.frames = sorted(frames)
        self._cache: dict[Path, Frame] = {}

    @classmethod
    def load(cls, index: Path | str) -> Recording:
        index = Path(index)
        rows = list(csv.DictReader(index.open(encoding="utf-8-sig")))
        frames = []
        for row in rows:
            t = row.get("source_time_s") or row.get("time_s")
            image = row.get("frame_image") or row.get("path")
            if t is None or image is None:
                raise ValueError(f"{index}: needs source_time_s and frame_image columns")
            frames.append((float(t), (index.parent / image).resolve()))
        return cls(frames)

    def shots(self, at: Sequence[Any], *, start_s: float | None = None) -> list[Shot]:
        """The scene's frames in play order; scene time starts at ``start_s`` (default: the first part's start)."""
        parts = segments(at)
        t0 = parts[0][0] if start_s is None else start_s
        shots: list[Shot] = []
        offset = t0
        for a, b in parts:
            for source, path in self.frames:
                if a <= source < b:
                    shots.append(Shot(offset + (source - a), source, path))
            offset += b - a
        if not shots:
            raise ValueError(f"no recorded frames in {at}")
        return shots

    def image(self, path: Path) -> Frame:
        """The frame for an image (encoded once; stamped by the caller)."""
        if path not in self._cache:
            self._cache[path] = image_frame(path.read_bytes(), timestamp_s=0)
        return self._cache[path]


class SceneFrames:
    """A scene as a step's :class:`~..step.Frames`; see the module docstring."""

    def __init__(self, recording: Recording, at: Sequence[Any]) -> None:
        self.recording = recording
        self.shots = recording.shots(at)
        self.start_s = self.shots[0].t_s
        self._loop_t0: float | None = None
        self._served = 0

    # -- the scene clock ---------------------------------------------------------------

    def begin(self) -> None:
        """Start the scene now (on the running loop's clock)."""
        self._loop_t0 = clock.now()

    def now(self) -> float:
        """Scene time now."""
        if self._loop_t0 is None:
            raise RuntimeError("the scene has not begun")
        return self.start_s + (clock.now() - self._loop_t0)

    @property
    def end_s(self) -> float:
        """When the scene is over: one frame spacing after its last frame."""
        last = self.shots[-1].t_s
        spacing = last - self.shots[-2].t_s if len(self.shots) > 1 else 1.0
        return last + spacing

    def remaining_s(self) -> float:
        return max(0.0, self.end_s - self.now())

    def switch(self, at: Sequence[Any]) -> None:
        """From now on, play ``at`` instead of the rest of the scene (the person did what they were told)."""
        now = self.now()
        kept = [shot for shot in self.shots if shot.t_s <= now]
        self.shots = kept + self.recording.shots(at, start_s=now)

    # -- Frames ------------------------------------------------------------------------

    def _shot(self) -> tuple[int, Shot] | None:
        now = self.now()
        if now >= self.end_s:
            return None
        index = max((i for i, shot in enumerate(self.shots) if shot.t_s <= now), default=0)
        return index, self.shots[index]

    def _frame(self, index: int, shot: Shot) -> Frame:
        self._served += 1
        return replace(self.recording.image(shot.path), timestamp_us=clock.wall_us(), sequence=self._served)

    async def current(self, *, max_age_s: float = 2.0, timeout_s: float = 5.0) -> Frame | None:
        found = self._shot()
        return None if found is None else self._frame(*found)

    async def every(self, interval_s: float, *, timeout_s: float = 5.0) -> AsyncIterator[Frame]:
        last: int | None = None
        while True:
            found = self._shot()
            if found is None:
                await asyncio.Event().wait()  # the scene is over: nothing new, ever (the evaluation ends the step)
            assert found is not None  # noqa: S101 - narrowed for the type checker
            index, shot = found
            if index == last:  # no new frame yet: wait for the next one, as a live stream does
                following = self.shots[index + 1].t_s if index + 1 < len(self.shots) else self.end_s
                await asyncio.sleep(max(0.01, following - self.now()))
                continue
            last = index
            yield self._frame(index, shot)
            await asyncio.sleep(interval_s)
