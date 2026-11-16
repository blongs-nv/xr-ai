# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watch the camera until enough distinct frames confirm a check: :func:`judge` in a loop with an evidence window.

A check is visual verification and nothing more, unless it declares escalation guidance and the caller passes an
:class:`Escalation`: then, while the check is not completing and some of its recent evidence is undecidable, a
text model looks at what is going on (how long it has run, what the frames showed, why some could not be decided,
the facts so far) and, following the guidance, decides whether the person could resolve it, and with which
question. The answer becomes a fact for every later frame (typed, when the check declares ``needs``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from xr_ai_app import FrameStream, Role, StructuredOutputError, clock

from .judge import Check, Facts, Judgment, Label, judge


@dataclass(frozen=True, slots=True)
class Escalation:
    """How a check may ask the person: who writes the question, how it is asked, and how often.

    ``role`` is a text model (an LLM: it decides whether the check is stuck, writes the question, and reads answers
    into the check's ``needs``). ``ask`` speaks a question and returns the person's answer, or ``None`` when none
    came (``Run.ask``). The model is consulted at most every ``review_every_s`` seconds, and only while some of the
    recent evidence is undecidable: clear negatives are progress toward a step not yet done, not being stuck. The
    limits that keep it from nagging: at most ``max_questions`` questions, never the same one twice, at least
    ``min_gap_s`` apart. These calls are outside the per-frame loop (watching pauses while they run) and decide what
    to ask a person, so they reason by default (``thinking``; ignored by a model without a switch).
    """

    role: Role
    ask: Callable[[str], Awaitable[str | None]]
    review_every_s: float = 15.0
    max_questions: int = 2
    min_gap_s: float = 20.0
    thinking: bool | None = True


class _Decision(BaseModel):
    ask: bool = Field(description="whether to ask the person now")
    question: str | None = Field(default=None, description="one short spoken question, when ask is true")
    reason: str = Field(default="", description="why ask, or why keep watching")


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
    """What is known beyond the camera: the starting facts and what the person answered."""

    asked: tuple[str, ...] = ()
    """The questions asked so far, in order."""

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
    facts: Facts | None = None,
    escalation: Escalation | None = None,
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
    obscured view neither confirms nor resets the check. ``facts`` are what is known at the start (an app setting,
    as an instance of ``check.needs``). With ``escalation`` and ``check.escalation`` guidance the check may ask the
    person when it gets stuck (see :class:`Escalation`); watching pauses while it waits for the answer. ``on_state``
    is awaited after every judged frame in this task; its errors propagate. Cancelling the task (a skip, a stop)
    ends the check with ``CancelledError``.
    """
    if not 1 <= required <= window:
        raise ValueError("required must be between 1 and window")
    if interval_s <= 0 or history < 1:
        raise ValueError("interval_s and history must be positive")
    started = clock.now()
    facts = facts or Facts()
    labels: deque[Label] = deque(maxlen=window)
    recent: deque[Judgment] = deque(maxlen=history)
    asked: list[str] = []
    last_asked: float | None = None
    last_review = started
    last: tuple[int, int] | None = None  # (capture time, sequence) of the last judged frame
    can_ask = escalation is not None and bool(check.escalation)
    async for frame in frames.every(interval_s, timeout_s=timeout_s or 5.0):
        if (frame.timestamp_us, frame.sequence) == last:
            continue
        last = (frame.timestamp_us, frame.sequence)
        judgment = await judge(frame, check, observe=observe, decide=decide, facts=facts, timeout_s=timeout_s)
        recent.append(judgment)
        if judgment.label != "unknown":
            labels.append(judgment.label)
        positives = sum(1 for label in labels if label == "positive")
        done = positives >= required
        now = clock.now()
        if (
            can_ask
            and not done
            and escalation is not None
            and now - last_review >= escalation.review_every_s
            and any(j.label == "unknown" for j in list(recent)[-window:])
            and len(asked) < escalation.max_questions
            and (last_asked is None or now - last_asked >= escalation.min_gap_s)
        ):
            last_review = now
            facts, question = await _escalate(check, escalation, tuple(recent), facts, asked, now - started, timeout_s)
            if question is not None:
                asked.append(question)
                last_asked = clock.now()
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
            tuple(asked),
        )
        if on_state is not None:
            await on_state(state)
        if done:
            return state
    raise RuntimeError("the frame stream ended before the check completed")  # the participant left


async def escalation_question(
    check: Check,
    role: Role,
    recent: Sequence[Judgment],
    facts: Facts,
    asked: Sequence[str] = (),
    elapsed_s: float | None = None,
    timeout_s: float | None = None,
    thinking: bool | None = True,
) -> str | None:
    """Whether the check is stuck on something the person could resolve, and the question; asks nothing.

    ``role`` (a text model) reads what is going on: the check, its escalation guidance, how long it has run, the
    recent observations (with why some could not be decided), the facts and the questions already asked. Returns
    the one question to ask now, or ``None`` to keep watching (also when the check has no guidance, when the model
    cannot answer, or when it proposes a question already asked). :func:`verify` asks what this returns; an
    evaluation can call it alone.
    """
    if not check.escalation:
        return None
    counts = {label: sum(1 for j in recent if j.label == label) for label in ("positive", "negative", "unknown")}
    running = f" for {elapsed_s:.0f} seconds" if elapsed_s is not None else ""
    lines = [
        f"A visual check has been running{running} and has not completed.",
        f"What it verifies: {check.criteria.instructions}",
        f"Done when: {check.criteria.positive}",
        f"Guidance on asking the person: {check.escalation}",
        f"Recent frames: {counts['positive']} positive, {counts['negative']} negative, {counts['unknown']} could not "
        "be decided. Oldest first (label, why it could not be decided, what was seen):",
        *(f"- {j.label}{f' ({j.reason})' if j.reason else ''}: {j.description[:300]}" for j in recent),
    ]
    if facts.text():
        lines.append(f"Already known:\n{facts.text()}")
    if asked:
        lines.append("Questions already asked: " + " | ".join(asked))
    lines.append(
        "Is the check stuck on something the person could resolve, following the guidance? If so, give the one "
        "short spoken question to ask now; if the camera is likely to settle it, keep watching."
    )
    try:
        decision = await role.ask(
            None, "\n".join(lines), response_format=_Decision, thinking=thinking, timeout_s=timeout_s
        )
    except (StructuredOutputError, TimeoutError):
        return None
    question = (decision.question or "").strip()
    if not decision.ask or not question or question in asked:
        return None
    return question


async def _escalate(
    check: Check,
    escalation: Escalation,
    recent: tuple[Judgment, ...],
    facts: Facts,
    asked: list[str],
    elapsed_s: float,
    timeout_s: float | None,
) -> tuple[Facts, str | None]:
    """Ask the person one question when the guidance calls for it; the facts with the answer, and the question."""
    question = await escalation_question(
        check, escalation.role, recent, facts, asked, elapsed_s, timeout_s, escalation.thinking
    )
    if question is None:
        return facts, None
    answer = await escalation.ask(question)
    if answer is None:
        return facts.with_answer(f'Asked the person: "{question}" No answer came.'), question
    known = None
    if check.needs is not None:
        prompt = (
            f'The assistant asked: "{question}" The person answered: "{answer}". Fill in only what the answer '
            "states; leave everything else null. 'I don't know' states nothing."
        )
        try:
            known = await escalation.role.ask(
                None, prompt, response_format=check.needs, thinking=escalation.thinking, timeout_s=timeout_s
            )
        except (StructuredOutputError, TimeoutError):
            known = None
    return facts.with_answer(f'Asked the person: "{question}" They answered: "{answer}"', known), question
