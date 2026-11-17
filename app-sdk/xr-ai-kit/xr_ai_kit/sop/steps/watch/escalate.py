# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The side escalation: an agent attached to one step, which helps when the step's fast check cannot decide.

An :class:`Escalator` is wired per step, where the step is built (``WatchStep.escalating(escalator)``); a step
without one never reasons. It carries everything about the escalation: its guidance (how the person can help, from
the step's words), its reasoning model, which context it reads (:class:`EscalationSettings` ``context``: nothing
beyond the step, the recent dialogue, the run's summary, any of the runner's channels by name), how often it looks
and how much it may say.

While the step runs, the escalator reads the fast check's recent reports (each frame's observation and label); when
most of the latest are ``unknown`` (:func:`stuck`), a reasoning model reads them with the check's criteria and the
facts so far (:func:`escalation_for`) and infers what is missing and whether the person can help: by telling it (a
question, whose answer goes into the step's :class:`~.judge.StepContext`, which the check reads on every frame) or
by doing something, such as bringing the kettle back into view (an instruction, spoken only). It sees no frames:
the fast check already looked. It never completes, skips or slows the step, and it ends (withdrawing a pending
question) when the step ends.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_app import Role, StructuredOutputError, clock

from .judge import Check, Facts, Judgment, StepContext

if TYPE_CHECKING:
    from ...step import StepIO


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
    guidance: str,
    facts: Facts | None = None,
    situation: str = "",
    asked: Sequence[str] = (),
    rule: tuple[int, int] = STUCK,
    thinking: bool | None = True,
    timeout_s: float | None = None,
) -> Escalation | None:
    """One judgement over the check's recent ``reports``: what to say to the person now, or ``None``.

    ``role`` is a reasoning text model. It reads what the check verifies and when it cannot decide, ``guidance``
    (how the person can help), the check's recent reports, ``situation`` (what is going on), ``facts`` and the lines
    already said (``asked``). ``None`` without guidance, while the check is not :func:`stuck`, on an
    unreadable reply, or for a line already said. Asks nobody: an evaluation calls it on recorded reports.
    """
    if not guidance or not stuck(reports, rule):
        return None
    facts = facts or Facts()
    lines = [
        "A fast camera check is verifying one step of a guided procedure, from the person's first-person view.",
        f"What it verifies: {check.criteria.instructions}",
        f"It completes when: {check.criteria.positive}",
        f"It cannot decide when: {check.criteria.unknown}",
        f"How the person can help: {guidance}",
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


class EscalationSettings(BaseModel):
    """How one step's escalator works (an app's settings may embed this, per step or for all)."""

    model_config = ConfigDict(extra="forbid")
    context: list[str] = []
    """What it reads besides the step's own status, in order: ``run`` (the run's summary: plan, steps' states, noted
    values) and the names of the runner's ``step_context`` channels (``dialogue``). Empty: the step alone."""
    every_s: float = 20.0
    """How often it looks at the check's reports."""
    max_lines: int = 2
    """How many things it may say during one step."""
    min_gap_s: float = 30.0
    """At least this long between two things it says."""
    stuck: tuple[int, int] = STUCK
    """When the check counts as stuck: this many ``unknown`` among its last so many reports."""
    thinking: bool | None = True


@dataclass(frozen=True, slots=True)
class Watched:
    """What a step offers its escalator: the check, its context, its latest reports, since when it runs."""

    step_id: str
    check: Check
    context: StepContext
    reports: Callable[[], Sequence[Judgment]]
    started_s: float


@dataclass(frozen=True, slots=True)
class Escalator:
    """One step's side escalation; see the module docstring.

    ``guidance`` says how the person can help (the step's ``escalation`` words); ``role`` reasons over the check's
    reports; ``reader`` (a text model; default ``role``) reads an answer into the check's ``needs``.
    """

    guidance: str
    role: Role
    reader: Role | None = None
    settings: EscalationSettings = field(default_factory=EscalationSettings)

    def but(self, **changes: object) -> Escalator:
        """A copy with some settings changed (``escalator.but(context=["dialogue"])``)."""
        settings = self.settings.model_validate({**self.settings.model_dump(), **changes})
        return replace(self, settings=settings)

    async def situation(self, watched: Watched, io: StepIO) -> str:
        """What it reads about what is going on: the step's own status, then its ``context``."""
        lines = [f"The step '{watched.step_id}' has been running for {clock.now() - watched.started_s:.0f} s."]
        if self.settings.context and (around := await io.situation(self.settings.context)):
            lines.append(around)
        return "\n".join(lines)

    async def run(self, watched: Watched, io: StepIO) -> None:
        """Run until cancelled: every ``every_s`` seconds, if the check is stuck, judge what to say; add answers.

        At most ``max_lines`` lines, never the same one twice, at least ``min_gap_s`` apart. A question is asked
        with ``io.ask`` and its answer added to the step's context; an instruction is spoken with ``io.say``.
        """
        settings, context = self.settings, watched.context
        asked: list[str] = []
        last_said: float | None = None
        while len(asked) < settings.max_lines:
            await asyncio.sleep(settings.every_s)
            if last_said is not None and clock.now() - last_said < settings.min_gap_s:
                continue
            recent = list(watched.reports())
            if not stuck(recent, settings.stuck):
                continue  # the check is deciding: nothing is missing
            said = await escalation_for(
                watched.check,
                self.role,
                recent,
                guidance=self.guidance,
                facts=context.facts,
                situation=await self.situation(watched, io),
                asked=asked,
                rule=settings.stuck,
                thinking=settings.thinking,
            )
            if said is None:
                continue
            asked.append(said.text)
            last_said = clock.now()
            if said.kind == "tell":
                await io.say(said.text)
                continue
            question = said.text
            answer = await io.ask(question)
            if answer is None:
                context.add(f'Asked the person: "{question}" No answer came.')
                continue
            known = await _read(watched.check, self.reader or self.role, question, answer)
            context.add(f'Asked the person: "{question}" They answered: "{answer}"', known)


async def _read(check: Check, reader: Role, question: str, answer: str) -> BaseModel | None:
    """The answer read into the check's ``needs`` (``None`` without needs, or when it states nothing readable)."""
    if check.needs is None:
        return None
    prompt = (
        f'The assistant asked: "{question}" The person answered: "{answer}". Fill in only what the answer states and '
        "leave everything else null; 'I don't know' states nothing."
    )
    try:
        return await reader.ask(None, prompt, response_format=check.needs)
    except (StructuredOutputError, TimeoutError):
        return None
