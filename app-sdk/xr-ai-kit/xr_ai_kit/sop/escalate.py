# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Side escalation: gather what a check cannot establish by asking the person, outside the fast loop.

:func:`gather` runs beside a step (``run.step(..., alongside=[...])``). It reads the fast check's recent reports (each
frame's observation and label); when most of its latest ones are ``unknown``, the check is missing something, and every
``every_s`` seconds a reasoning model reads those reports with the check's criteria and the facts so far and infers
what is missing and whether the person can help: by telling it (a question, whose answer goes into the step's
:class:`~.judge.StepContext`, which the check reads on every frame) or by doing something, such as bringing the
kettle back into view (an instruction, spoken only). It sees no frames: the fast check already looked. It never
completes, skips or slows the step, and it ends (withdrawing a pending question) when the step ends.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from xr_ai_app import Role, StructuredOutputError

from .judge import Check, Facts, Judgment, StepContext


class _Decision(BaseModel):
    missing: str = Field(default="", description="what the check is missing, in a few words; empty when nothing")
    action: Literal["none", "ask", "tell"] = Field(
        description="ask: a question the person answers; tell: something for the person to do; none: nothing now"
    )
    words: str | None = Field(default=None, description="for ask or tell: one short spoken line")


@dataclass(frozen=True, slots=True)
class Escalation:
    """What the side escalation says to the person: a question to answer, or an instruction to follow."""

    kind: Literal["ask", "tell"]
    text: str


STUCK = (3, 4)
"""Stuck: at least 3 of the check's last 4 reports are ``unknown``."""


def stuck(reports: Sequence[Judgment], rule: tuple[int, int] = STUCK) -> bool:
    """Whether the check has mostly stopped deciding: ``rule[0]`` of its last ``rule[1]`` reports are ``unknown``.

    One unreadable frame among decisions is noise; mostly unknowns, even with a stray misread among them (a number
    read off a display cut off by the edge of the view), mean the check is missing something.
    """
    unknowns, of = rule
    recent = reports[-of:]
    return len(recent) >= of and sum(r.label == "unknown" for r in recent) >= unknowns


async def escalation_for(
    check: Check,
    role: Role,
    reports: Sequence[Judgment],
    *,
    facts: Facts | None = None,
    situation: str = "",
    asked: Sequence[str] = (),
    thinking: bool | None = True,
    timeout_s: float | None = None,
) -> Escalation | None:
    """One judgement over the check's recent ``reports``: what to say to the person now, or ``None``.

    ``role`` is a reasoning text model. It reads what the check verifies and when it cannot decide, the check's
    escalation guidance, its recent reports, ``situation`` (the app's account of what is going on), ``facts`` and the
    lines already said (``asked``). ``None`` without guidance, while the check is not :func:`stuck`, on an
    unreadable reply, or for a line already said. Asks nobody: an evaluation calls it on recorded reports.
    """
    if not check.escalation or not stuck(reports):
        return None
    facts = facts or Facts()
    lines = [
        "A fast camera check is verifying one step of a guided procedure, from the person's first-person view.",
        f"What it verifies: {check.criteria.instructions}",
        f"It completes when: {check.criteria.positive}",
        f"It cannot decide when: {check.criteria.unknown}",
        f"How the person can help: {check.escalation}",
        "Its recent reports, oldest first (what it observed in each frame, and its label):",
        *(f"- {r.label}: {r.description[:300]}" + (f" ({r.reason})" if r.reason else "") for r in reports),
    ]
    if situation:
        lines.append(f"What is going on:\n{situation}")
    lines.append(f"Already known:\n{facts.text()}" if facts.text() else "Nothing has been learned from the person yet.")
    if asked:
        lines.append("Already said to the person: " + " | ".join(asked))
    lines.append(
        "From the reports, what is the check missing that keeps it from deciding? Reports with nothing readable "
        "(no reading, nothing of the item seen) mean the camera does not see what it checks: the person is looking "
        "elsewhere or something is in the way, so tell them what to bring into view; the check then reads it "
        "itself. Reports that see the item clearly but still cannot decide mean the check needs something the "
        "person knows: ask for that, unless it is already known. Give one short spoken line, or none when the "
        "person cannot help."
    )
    try:
        decision = await role.ask(
            None, "\n".join(lines), response_format=_Decision, thinking=thinking, timeout_s=timeout_s
        )
    except (StructuredOutputError, TimeoutError):
        return None
    text = (decision.words or "").strip()
    if decision.action == "none" or not text or text in asked:
        return None
    return Escalation(decision.action, text)


async def gather(
    context: StepContext,
    check: Check,
    reports: Callable[[], Sequence[Judgment]],
    *,
    role: Role,
    ask: Callable[[str], Awaitable[str | None]],
    say: Callable[[str], Awaitable[None]] | None = None,
    reader: Role | None = None,
    situation: Callable[[], Awaitable[str]] | None = None,
    every_s: float = 20.0,
    max_questions: int = 2,
    min_gap_s: float = 30.0,
    thinking: bool | None = True,
) -> None:
    """Run until cancelled: every ``every_s`` seconds, if the check is stuck, judge what to say; add answers.

    ``reports`` returns the check's recent judgments (for example the latest :class:`~.verify.Verification`'s). ``ask``
    speaks a question and returns the person's answer, or ``None`` (``Run.ask``); ``say`` speaks an instruction
    (``Run.say``; without it instructions are not given). ``reader`` (a text model) reads an
    answer into the check's ``needs``; without it the answer is kept as text. ``situation`` returns what is going on
    (for example the guide's state and the recent conversation). At most ``max_questions`` lines, never the same one
    twice, at least ``min_gap_s`` apart. A check without escalation guidance makes this return at once.
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
        recent = list(reports())
        if not stuck(recent):
            continue  # the check is deciding: nothing is missing
        text = await situation() if situation is not None else ""
        said = await escalation_for(
            check, role, recent, facts=context.facts, situation=text, asked=asked, thinking=thinking
        )
        if said is None or (said.kind == "tell" and say is None):
            continue
        asked.append(said.text)
        last_asked = loop.time()
        if said.kind == "tell":
            await say(said.text)
            continue
        question = said.text
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
