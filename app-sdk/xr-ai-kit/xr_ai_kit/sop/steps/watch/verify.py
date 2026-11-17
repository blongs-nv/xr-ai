# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watch the camera until enough distinct frames confirm a check: :func:`judge` in a loop with an evidence window.

A fast loop and nothing else: no reasoning, no questions. What the camera cannot establish comes in through the
step's :class:`~.judge.StepContext`, which the loop reads on every frame and never writes (see :mod:`.escalator` for a
side task that fills it by asking the person).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from xr_ai_app import FrameStream, Role, clock

from .judge import Check, Facts, Judgment, Label, StepContext, judge


@dataclass(frozen=True, slots=True)
class Verification:
    """The state of one visual check, reported after every judged frame and returned when it completes."""

    status: Literal["running", "completed"]
    question: str
    judgments: tuple[Judgment, ...]
    """The most recent judgments, oldest first (at most ``history`` of them), unknowns included."""

    window: tuple[Label, ...]
    """The decidable labels counted toward completion, oldest first (unknowns are not in it)."""

    positives: int
    required: int
    window_size: int
    started_s: float
    updated_s: float
    """Times on the core clock (``xr_ai_app.clock.now``)."""

    facts: Facts = field(default_factory=Facts)
    """The context the last frame was judged with (from the step's context channel)."""

    def summary(self) -> str:
        """One line for context and logs: progress toward completion, the latest observation, the facts."""
        latest = self.judgments[-1] if self.judgments else None
        seen = f" Latest: {latest.label}: {latest.description[:300]}" if latest else " No frame judged yet."
        known = f" {self.facts.text()}" if self.facts.text() else ""
        return (
            f"{self.status}: {self.positives} of {self.required} confirmations in the last {self.window_size}."
            f"{seen}{known}"
        )


async def verify(
    frames: FrameStream,
    check: Check,
    *,
    observe: Role,
    decide: Role,
    context: StepContext | None = None,
    required: int = 2,
    window: int = 4,
    interval_s: float = 2.0,
    history: int = 8,
    timeout_s: float | None = 15.0,
    on_state: Callable[[Verification], Awaitable[None]] | None = None,
) -> Verification:
    """Judge a frame every ``interval_s`` seconds until ``required`` of the last ``window`` decidable ones are positive.

    Frames come from ``frames.every`` (so a device with no video stream is asked for a picture on each tick), and a
    frame is never judged twice. ``unknown`` judgments are kept for context but take no slot in the window, so an
    obscured view neither confirms nor resets the check. Each frame is judged with ``context``'s facts as they are at
    that moment. ``on_state`` is awaited after every judged frame in this task; its errors propagate. Cancelling the
    task (a skip, a stop) ends the check with ``CancelledError``.
    """
    if not 1 <= required <= window:
        raise ValueError("required must be between 1 and window")
    if interval_s <= 0 or history < 1:
        raise ValueError("interval_s and history must be positive")
    started = clock.now()
    labels: deque[Label] = deque(maxlen=window)
    recent: deque[Judgment] = deque(maxlen=history)
    last: tuple[int, int] | None = None  # (capture time, sequence) of the last judged frame
    async for frame in frames.every(interval_s, timeout_s=timeout_s or 5.0):
        if (frame.timestamp_us, frame.sequence) == last:
            continue
        last = (frame.timestamp_us, frame.sequence)
        facts = context.facts if context is not None else Facts()
        judgment = await judge(frame, check, observe=observe, decide=decide, facts=facts, timeout_s=timeout_s)
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
            facts,
        )
        if on_state is not None:
            await on_state(state)
        if done:
            return state
    raise RuntimeError("the frame stream ended before the check completed")  # the participant left
