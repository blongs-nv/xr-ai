# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Side escalation: gather what a check cannot see by asking the person, outside the fast loop.

:func:`gather` runs beside a step (``run.step(..., alongside=[...])``). Every ``every_s`` seconds a reasoning model
looks at the latest frame and what is going on (the app's account of the situation, which includes the check's
latest report, the facts so far, the questions already asked) and judges, following the check's escalation
guidance, whether to ask the person one question. The answer goes into the step's :class:`~.judge.StepContext`,
which the check reads on every frame. That is its only effect: it never completes, skips or slows the step, and it
ends (withdrawing a pending question) when the step ends.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel, Field
from xr_ai_app import Frame, FrameStream, Role, StructuredOutputError

from .judge import Check, Facts, StepContext


class _Decision(BaseModel):
    ask: bool = Field(description="whether to ask the person now")
    question: str | None = Field(default=None, description="when ask is true: one short spoken question")
    reason: str = ""


async def question_for(
    check: Check,
    role: Role,
    frame: Frame | None,
    *,
    facts: Facts | None = None,
    situation: str = "",
    asked: Sequence[str] = (),
    thinking: bool | None = True,
    timeout_s: float | None = None,
) -> str | None:
    """One judgement: the question to ask the person now, or ``None``; asks nothing.

    ``role`` is a reasoning model (one that takes images when ``frame`` is given). It reads the check (what it
    verifies, its escalation guidance), the latest frame, ``situation`` (the app's account of what is going on),
    ``facts`` and the questions already ``asked``. ``None`` also without guidance, on an unreadable reply, or for a
    question already asked. Usable alone: an evaluation calls it on recorded frames.
    """
    if not check.escalation:
        return None
    facts = facts or Facts()
    lines = [
        "A camera check is verifying one step of a guided procedure, from the person's first-person view.",
        f"What it verifies: {check.criteria.instructions}",
        f"Done when: {check.criteria.positive}",
        f"What the camera may not establish, which the person can tell: {check.escalation}",
    ]
    if frame is not None:
        lines.append(
            "The image is the latest camera frame. Read it yourself: the check's reports below come from a fast "
            "vision model, which can misread or fill in what is not shown."
        )
    if situation:
        lines.append(f"What is going on:\n{situation}")
    if facts.text():
        lines.append(f"Already known:\n{facts.text()}")
    if asked:
        lines.append("Questions already asked: " + " | ".join(asked))
    lines.append(
        "Each question interrupts the person while they work, so one is worth asking only when the check needs "
        "something that neither the image nor what is already known establishes. Should the assistant ask the "
        "person one short question now? If so, give the question."
    )
    try:
        decision = await role.ask(
            frame, "\n".join(lines), response_format=_Decision, thinking=thinking, timeout_s=timeout_s
        )
    except (StructuredOutputError, TimeoutError):
        return None
    question = (decision.question or "").strip()
    if not decision.ask or not question or question in asked:
        return None
    return question


async def gather(
    context: StepContext,
    check: Check,
    frames: FrameStream,
    *,
    role: Role,
    ask: Callable[[str], Awaitable[str | None]],
    reader: Role | None = None,
    situation: Callable[[], Awaitable[str]] | None = None,
    every_s: float = 20.0,
    max_questions: int = 2,
    min_gap_s: float = 30.0,
    thinking: bool | None = True,
) -> None:
    """Run until cancelled: judge every ``every_s`` seconds whether to ask, ask, and add the answer to ``context``.

    ``ask`` speaks a question and returns the person's answer, or ``None`` (``Run.ask``). ``reader`` (a text model)
    reads an answer into the check's ``needs``; without it the answer is kept as text. ``situation`` returns what is
    going on (for example the guide's summary, which includes the check's latest report, and the recent
    conversation). At most ``max_questions`` questions, never the same one twice, at least ``min_gap_s`` apart. A
    check without escalation guidance makes this return at once.
    """
    if not check.escalation:
        return
    asked: list[str] = []
    last_asked: float | None = None
    loop = asyncio.get_running_loop()
    while len(asked) < max_questions:
        await asyncio.sleep(every_s)
        if last_asked is not None and loop.time() - last_asked < min_gap_s:
            continue
        frame = await frames.current(timeout_s=5.0)
        text = await situation() if situation is not None else ""
        question = await question_for(
            check, role, frame, facts=context.facts, situation=text, asked=asked, thinking=thinking
        )
        if question is None:
            continue
        asked.append(question)
        last_asked = loop.time()
        answer = await ask(question)
        if answer is None:
            context.add(f'Asked the person: "{question}" No answer came.')
            continue
        context.add(
            f'Asked the person: "{question}" They answered: "{answer}"', await _read(check, reader, question, answer)
        )


async def _read(check: Check, reader: Role | None, question: str, answer: str) -> BaseModel | None:
    """The answer read into the check's ``needs`` (``None`` without needs, a reader, or anything stated)."""
    if check.needs is None or reader is None:
        return None
    prompt = (
        f'The assistant asked: "{question}" The person answered: "{answer}". Fill in only what the answer states and '
        "leave everything else null; 'I don't know' states nothing."
    )
    try:
        return await reader.ask(None, prompt, response_format=check.needs)
    except (StructuredOutputError, TimeoutError):
        return None
