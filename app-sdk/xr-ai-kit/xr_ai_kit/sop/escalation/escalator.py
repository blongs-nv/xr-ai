# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The escalator: an agent of its own, beside one step, that helps the person when the step cannot decide.

It knows the step only through the run's :class:`~.link.Link`: it reads the step's outbound context (what it does,
its recent reports, what it knows) and gives it inbound context (what the person said). It talks to the person
itself, through its :class:`~.link.EscalationIO`, never through the step, and it never completes, skips or slows
the step. Which step it is attached to, and how, is :class:`~.escalated.Escalated`.

While the step runs it looks at the step's reports every ``every_s``; when most of the latest are undecided
(:func:`stuck`), a reasoning model reads them with the step's task, ``guidance`` (how the person can help) and what
is known (:func:`escalation_for`) and infers what is missing and whether the person can help: by telling it (a
question, whose answer goes inbound) or by doing something, such as bringing the kettle back into view (an
instruction, spoken only). It sees no frames: the step already looked.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_app import Role, StructuredOutputError, clock

from ..steps.step_definitions import StepDefinition
from .link import EscalationIO, Inbound, Link, Outbound, Report


class _Decision(BaseModel):
    missing: str = Field(default="", description="what the check is missing, in a few words; empty when nothing")
    action: Literal["none", "ask", "tell"] = Field(
        description="ask: a question the person answers; tell: something for the person to do; none: nothing now"
    )
    words: str | None = Field(default=None, description="for ask or tell: one short spoken line")


@dataclass(frozen=True, slots=True)
class Escalation:
    """What the escalator says to the person: a question to answer, or an instruction to follow."""

    kind: Literal["ask", "tell"]
    text: str


STUCK = (3, 4)
"""Stuck: at least 3 of the step's last 4 reports are undecided."""


def stuck(reports: Sequence[Report], rule: tuple[int, int] = STUCK) -> bool:
    """Whether the step has mostly stopped deciding: ``rule[0]`` of its last ``rule[1]`` reports are undecided.

    One undecided report among decisions is noise; mostly undecided ones, even with a stray misread among them (a
    number read off a display cut off by the edge of the view), mean the step is missing something.
    """
    undecided, of = rule
    recent = reports[-of:]
    return len(recent) >= of and sum(not r.decided for r in recent) >= undecided


async def escalation_for(
    step: Outbound,
    role: Role,
    *,
    guidance: str,
    situation: str = "",
    asked: Sequence[str] = (),
    rule: tuple[int, int] = STUCK,
    thinking: bool | None = True,
    timeout_s: float | None = None,
) -> Escalation | None:
    """One judgement over the step's outbound context: what to say to the person now, or ``None``.

    ``role`` is a reasoning text model. It reads the step's task, ``guidance`` (how the person can help), the step's
    recent reports, ``situation`` (what is going on), what the step knows and the lines already said (``asked``).
    ``None`` without guidance, while the step is not :func:`stuck`, on an unreadable reply, or for a line already
    said. Asks nobody: an evaluation calls it on recorded reports.
    """
    if not guidance or not stuck(step.reports, rule):
        return None
    lines = [
        *step.task,
        f"How the person can help: {guidance}",
        "Its recent reports, oldest first (what it observed in each frame, and its label):",
        *(f"- {r.label}: {r.text}" for r in step.reports),
    ]
    if situation:
        lines.append(f"What is going on:\n{situation}")
    lines.append(f"Already known:\n{step.known}" if step.known else "Nothing has been learned from the person yet.")
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
    """How one escalator works (an app's settings may embed this, per step or for all)."""

    model_config = ConfigDict(extra="forbid")
    context: list[str] = []
    """What it reads besides the step's outbound context, in order: ``run`` (the run's summary: plan, steps' states,
    noted values) and the names of the runner's ``escalation_context`` channels (``dialogue``). Empty: the step
    alone."""
    every_s: float = 20.0
    """How often it looks at the step's reports."""
    max_lines: int = 2
    """How many things it may say during one step."""
    min_gap_s: float = 30.0
    """At least this long between two things it says."""
    stuck: tuple[int, int] = STUCK
    """When the step counts as stuck: this many undecided among its last so many reports."""
    thinking: bool | None = True


@dataclass(frozen=True, slots=True)
class Escalator:
    """One step's escalator; see the module docstring.

    ``guidance`` says how the person can help (the step's ``escalation`` words); ``role`` reasons over the step's
    reports; ``reader`` (a text model; default ``role``) reads an answer into the typed facts the step accepts.
    """

    guidance: str
    role: Role
    reader: Role | None = None
    settings: EscalationSettings = field(default_factory=EscalationSettings)

    @classmethod
    def define(
        cls,
        definition: StepDefinition,
        role: Role,
        settings: EscalationSettings | None = None,
        *,
        reader: Role | None = None,
    ) -> Escalator:
        """The escalator for a step, its guidance the ``escalation`` section of the step's definition."""
        guidance = definition.section("escalation", str)
        return cls(guidance, role, reader, settings or EscalationSettings())

    def but(self, **changes: object) -> Escalator:
        """A copy with some settings changed (``escalator.but(context=["dialogue"])``)."""
        settings = self.settings.model_validate({**self.settings.model_dump(), **changes})
        return replace(self, settings=settings)

    async def situation(self, step: Outbound, io: EscalationIO) -> str:
        """What it reads about what is going on: how long the step has run, then its ``context``."""
        lines = [f"The step '{step.step_id}' has been running for {clock.now() - step.started_s:.0f} s."]
        if self.settings.context and (around := await io.situation(self.settings.context)):
            lines.append(around)
        return "\n".join(lines)

    async def run(self, link: Link, io: EscalationIO) -> None:
        """Run until cancelled: every ``every_s`` seconds, if the step is stuck, judge what to say; send answers.

        At most ``max_lines`` lines, never the same one twice, at least ``min_gap_s`` apart. A question is asked
        with ``io.ask`` and its answer sent inbound; an instruction is spoken with ``io.say``.
        """
        settings = self.settings
        asked: list[str] = []
        last_said: float | None = None
        while len(asked) < settings.max_lines:
            await asyncio.sleep(settings.every_s)
            if last_said is not None and clock.now() - last_said < settings.min_gap_s:
                continue
            step = link.outbound()
            if step is None or not stuck(step.reports, settings.stuck):
                continue  # not started, or deciding: nothing is missing
            said = await escalation_for(
                step,
                self.role,
                guidance=self.guidance,
                situation=await self.situation(step, io),
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
                link.inbound(Inbound(f'Asked the person: "{question}" No answer came.'))
                continue
            known = await _read(step.needs, self.reader or self.role, question, answer)
            link.inbound(Inbound(f'Asked the person: "{question}" They answered: "{answer}"', known))


async def _read(needs: type[BaseModel] | None, reader: Role, question: str, answer: str) -> BaseModel | None:
    """The answer read into ``needs`` (``None`` without needs, or when it states nothing readable)."""
    if needs is None:
        return None
    prompt = (
        f'The assistant asked: "{question}" The person answered: "{answer}". Fill in only what the answer states and '
        "leave everything else null; 'I don't know' states nothing."
    )
    try:
        return await reader.ask(None, prompt, response_format=needs)
    except (StructuredOutputError, TimeoutError):
        return None
