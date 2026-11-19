# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watch the camera until enough distinct frames confirm a check: :func:`judge` in a loop with an evidence window.

A fast loop and nothing else: no reasoning, no questions. What the camera cannot establish comes in through the
step's :class:`~.judge.StepKnowledge`, which the loop reads on every frame and never writes (the escalator beside the
step, in ``...escalation``, adds what the person answers or says).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict
from xr_ai_app import FrameStream, clock

from .judge import Check, Judgment, Knowledge, Label, StepKnowledge, WatchModels, judge

DESCRIPTION_CHARS = 300
"""How much of a frame's description a summary or an escalator's evidence shows."""


class WatchSettings(BaseModel):
    """How a watch step watches (an app's settings may embed this)."""

    model_config = ConfigDict(extra="forbid")
    required: int = 2
    """Positives needed ..."""
    window: int = 4
    """... among the last this many decidable frames."""
    interval_s: float = 2.0
    """How often a frame is judged."""
    model_timeout_s: float = 30.0
    """Bound on one model call."""
    frame_timeout_s: float = 5.0
    """How long to wait for a picture from a camera that takes one on request."""


@dataclass(frozen=True, slots=True)
class Verification:
    """The state of one visual check, reported after every judged frame and returned when it completes."""

    status: Literal["running", "completed"]
    question: str
    judgments: tuple[Judgment, ...]
    """The most recent judgments, oldest first (at most ``history`` of them), unknowns included."""

    decidable_labels: tuple[Label, ...]
    """The decidable labels counted toward completion, oldest first (unknowns are not among them)."""

    positives: int
    required: int
    window_size: int
    started_s: float
    updated_s: float
    """Times on the core clock (``xr_ai_app.clock.now``)."""

    knowledge: Knowledge = field(default_factory=Knowledge)
    """What the step knew beyond the camera when the last frame was judged."""

    def summary(self) -> str:
        """One line for context and logs: progress toward completion, the latest observation, the knowledge."""
        latest = self.judgments[-1] if self.judgments else None
        seen = (
            f" Latest: {latest.label}: {latest.description[:DESCRIPTION_CHARS]}" if latest else " No frame judged yet."
        )
        known = self.knowledge.text()
        return (
            f"{self.status}: {self.positives} of {self.required} confirmations in the last {self.window_size}."
            f"{seen}{f' {known}' if known else ''}"
        )


async def verify(
    frames: FrameStream,
    check: Check,
    models: WatchModels,
    settings: WatchSettings,
    *,
    knowledge: StepKnowledge | None = None,
    on_state: Callable[[Verification], Awaitable[None]] | None = None,
    history: int = 8,
) -> Verification:
    """Judge a frame every ``settings.interval_s`` seconds until ``settings.required`` of the last ``settings.window``
    decidable ones are positive.

    Frames come from ``frames.every`` (so a device with no video stream is asked for a picture on each tick, waiting
    at most ``frame_timeout_s``), and a frame is never judged twice. ``unknown`` judgments are kept for context but
    take no slot in the window, so an obscured view neither confirms nor resets the check. Each frame is judged with
    ``knowledge`` as it is at that moment. The last ``history`` judgments are kept (not a setting: an escalator reads
    them). ``on_state`` is awaited after every judged frame in this task; its errors propagate. Cancelling the task (a
    skip, a stop) ends the check with ``CancelledError``.
    """
    required, window = settings.required, settings.window
    if not 1 <= required <= window:
        raise ValueError("required must be between 1 and window")
    if settings.interval_s <= 0 or history < 1:
        raise ValueError("interval_s and history must be positive")
    started = clock.now()
    labels: deque[Label] = deque(maxlen=window)
    recent: deque[Judgment] = deque(maxlen=history)
    last: tuple[int, int] | None = None  # (capture time, sequence) of the last judged frame
    async for frame in frames.every(settings.interval_s, timeout_s=settings.frame_timeout_s):
        if (frame.timestamp_us, frame.sequence) == last:
            continue  # defensive: a custom Frames may hand the same frame twice
        last = (frame.timestamp_us, frame.sequence)
        known = knowledge.knowledge if knowledge is not None else Knowledge()
        judgment = await judge(frame, check, models, knowledge=known, timeout_s=settings.model_timeout_s)
        recent.append(judgment)
        if judgment.label != "unknown":
            labels.append(judgment.label)
        positives = sum(1 for label in labels if label == "positive")
        done = positives >= required
        state = Verification(
            "completed" if done else "running",
            check.question,
            tuple(recent),
            tuple(labels),
            positives,
            required,
            window,
            started,
            clock.now(),
            known,
        )
        if on_state is not None:
            await on_state(state)
        if done:
            return state
    raise RuntimeError("the frame stream ended before the check completed")  # the participant left
