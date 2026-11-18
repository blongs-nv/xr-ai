# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kit's escalator (:class:`ReasoningEscalator`): an agent of its own, beside one step, that helps the person
when the step cannot decide, by reasoning in text over the step's evidence. (Any :class:`~.escalation_channel.Escalator`
can stand in its place: one that sends the frame behind a piece of evidence to a model of its own, say.)

It knows the step only through the run's :class:`~.escalation_channel.EscalationChannel`: it reads the step's
snapshot (what it does, its recent evidence, what it knows) and adds context to it (what the person said). It talks
to the person itself, through its :class:`~.escalation_channel.EscalatorEnvironment`, never through the step, and it
never completes, skips or slows the step. Which step it is beside is :class:`~.step_with_escalator.StepWithEscalator`.

While the step runs it looks at the step's evidence every ``every_s``; when most of the latest is undecided
(:func:`is_stuck`), a reasoning model reads it with the step's task, ``guidance`` (how the person can help) and what
is known (:func:`decide_escalation`) and infers what is missing and whether the person can help: by telling it (a
question, whose answer is added to the step's context) or by doing something, such as bringing the kettle back into
view (an instruction, spoken only). It sees no frames: the step already looked.

Its prompt is per step: the step's ``escalation`` section gives the guidance and may replace the closing
``instructions`` or the whole ``prompt`` (:data:`PROMPT`, a template; see :class:`EscalationSection`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from xr_ai_app import Role, StructuredOutputError, clock

from ..steps.step_definitions import StepDefinition
from .escalation_channel import AddedContext, EscalationChannel, EscalatorEnvironment, Evidence, StepSnapshot


class _Decision(BaseModel):
    missing: str = Field(default="", description="what the check is missing, in a few words; empty when nothing")
    action: Literal["none", "ask", "tell"] = Field(
        description="ask: a question the person answers; tell: something for the person to do; none: nothing now"
    )
    words: str | None = Field(default=None, description="for ask or tell: one short spoken line")


@dataclass(frozen=True, slots=True)
class EscalationMessage:
    """What the escalator says to the person: a question to answer, or an instruction to follow."""

    kind: Literal["ask", "tell"]
    text: str


STUCK_RULE = (3, 4)
"""Stuck: at least 3 of the step's last 4 pieces of evidence are undecided."""


def is_stuck(evidence: Sequence[Evidence], rule: tuple[int, int] = STUCK_RULE) -> bool:
    """Whether the step has mostly stopped deciding: ``rule[0]`` of its last ``rule[1]`` pieces of evidence are
    undecided.

    One undecided piece among decisions is noise; mostly undecided ones, even with a stray misread among them (a
    number read off a display cut off by the edge of the view), mean the step is missing something.
    """
    undecided, of = rule
    recent = evidence[-of:]
    return len(recent) >= of and sum(not e.decided for e in recent) >= undecided


INSTRUCTIONS = (
    "From the reports, what is the check missing that keeps it from deciding? Reports with nothing readable "
    "(no reading, nothing of the item seen) mean the camera does not see what it checks: the person is looking "
    "elsewhere or something is in the way, so tell them what to bring into view; the check then reads it "
    "itself. Reports that see the item clearly but still cannot decide mean the check needs something the "
    "person knows: ask for that, unless it is already known. Give one short spoken line, or none when the "
    "person cannot help."
)
"""What the reasoning model is asked to do, last in the prompt (a step may give its own)."""

PROMPT = (
    "{task}\n"
    "How the person can help: {guidance}\n"
    "Its recent reports, oldest first (what it observed in each frame, and its label):\n"
    "{evidence}{situation}\n"
    "{known}{asked}\n"
    "{instructions}"
)
"""The whole prompt, as a template: ``{task}`` (the step's lines saying what it does), ``{guidance}``,
``{evidence}`` (one ``- label: description`` line each), ``{situation}`` (``\nWhat is going on:\n...`` or empty),
``{known}`` (``Already known:\n...`` or that nothing has been learned), ``{asked}`` (``\nAlready said to the
person: ...`` or empty) and ``{instructions}``. Literal braces in a step's own template are doubled."""


async def decide_escalation(
    step: StepSnapshot,
    reason: Role,
    *,
    guidance: str,
    situation: str = "",
    asked: Sequence[str] = (),
    rule: tuple[int, int] = STUCK_RULE,
    thinking: bool | None = True,
    timeout_s: float | None = None,
    instructions: str = INSTRUCTIONS,
    prompt: str = PROMPT,
) -> EscalationMessage | None:
    """One judgement over a snapshot of the step: what to say to the person now, or ``None``.

    ``reason`` is a reasoning text model. It reads ``prompt`` filled in: the step's task, ``guidance`` (how the
    person can help), the step's recent evidence, ``situation`` (what is going on), what the step knows, the lines
    already said (``asked``) and ``instructions``. ``None`` without guidance, while the step is not :func:`is_stuck`,
    on an unreadable reply, or for a line already said. Asks nobody: an evaluation calls it on recorded evidence.
    """
    if not guidance or not is_stuck(step.evidence, rule):
        return None
    text = prompt.format(
        task="\n".join(step.task),
        guidance=guidance,
        evidence="\n".join(f"- {e.label}: {e.description}" for e in step.evidence),
        situation=f"\nWhat is going on:\n{situation}" if situation else "",
        known=f"Already known:\n{step.known}" if step.known else "Nothing has been learned from the person yet.",
        asked=("\nAlready said to the person: " + " | ".join(asked)) if asked else "",
        instructions=instructions,
    )
    try:
        decision = await reason.ask(None, text, response_format=_Decision, thinking=thinking, timeout_s=timeout_s)
    except (StructuredOutputError, TimeoutError):
        return None
    said = (decision.words or "").strip()
    if decision.action == "none" or not said or said in asked:
        return None
    return EscalationMessage(decision.action, said)


class EscalationSection(BaseModel):
    """A step's ``escalation`` section: its guidance alone (a string), or a mapping that may also replace the closing
    instructions or the whole prompt for this step."""

    model_config = ConfigDict(extra="forbid")
    guidance: str
    """How the person can help when the step cannot decide."""
    instructions: str = INSTRUCTIONS
    """What the reasoning model is asked to do (default :data:`INSTRUCTIONS`)."""
    prompt: str = PROMPT
    """The whole prompt (default :data:`PROMPT`)."""

    @model_validator(mode="before")
    @classmethod
    def _guidance_alone(cls, value: object) -> object:
        return {"guidance": value} if isinstance(value, str) else value


class EscalationSettings(BaseModel):
    """How one escalator works (an app's settings may embed this, per step or for all)."""

    model_config = ConfigDict(extra="forbid")
    context: list[str] = []
    """What it reads besides the step's snapshot, in order: ``run`` (the run's summary: plan, steps' states,
    noted values) and the names of the runner's ``escalation_context`` channels (``dialogue``). Empty: the step
    alone."""
    every_s: float = 20.0
    """How often it looks at the step's evidence."""
    max_lines: int = 2
    """How many things it may say during one step."""
    min_gap_s: float = 30.0
    """At least this long between two things it says."""
    stuck: tuple[int, int] = STUCK_RULE
    """When the step counts as stuck: this many undecided among its last so many pieces of evidence."""
    thinking: bool | None = True


class ReasoningEscalator:
    """The kit's escalator for one step; see the module docstring.

    Built from the step's definition: its ``escalation`` section (:class:`EscalationSection`) gives ``guidance`` (how
    the person can help) and this step's ``instructions`` and ``prompt``. ``reason`` reasons over the step's
    evidence; ``answer_reader`` (a text model; default ``reason``) reads an answer into the typed facts the step
    accepts.
    """

    def __init__(
        self,
        definition: StepDefinition,
        reason: Role,
        settings: EscalationSettings | None = None,
        *,
        answer_reader: Role | None = None,
    ) -> None:
        section = definition.section("escalation", EscalationSection)
        self.guidance = section.guidance
        self.instructions = section.instructions
        self.prompt = section.prompt
        self.reason = reason
        self.answer_reader = answer_reader or reason
        self.settings = settings or EscalationSettings()

    async def describe_situation(self, step: StepSnapshot, environment: EscalatorEnvironment) -> str:
        """What it reads about what is going on: how long the step has run, then its ``context``."""
        lines = [f"The step '{step.step_id}' has been running for {clock.now() - step.started_s:.0f} s."]
        if self.settings.context and (around := await environment.read_context(self.settings.context)):
            lines.append(around)
        return "\n".join(lines)

    async def run(self, channel: EscalationChannel, environment: EscalatorEnvironment) -> None:
        """Run until cancelled: every ``every_s`` seconds, if the step is stuck, judge what to say; pass answers on.

        At most ``max_lines`` lines, never the same one twice, at least ``min_gap_s`` apart. A question is asked
        with ``environment.ask`` and its answer added to the step's context; an instruction is spoken with
        ``environment.say``.
        """
        settings = self.settings
        asked: list[str] = []
        last_said: float | None = None
        while len(asked) < settings.max_lines:
            await asyncio.sleep(settings.every_s)
            if last_said is not None and clock.now() - last_said < settings.min_gap_s:
                continue
            step = channel.snapshot()
            if step is None or not is_stuck(step.evidence, settings.stuck):
                continue  # not started, or deciding: nothing is missing
            message = await decide_escalation(
                step,
                self.reason,
                guidance=self.guidance,
                situation=await self.describe_situation(step, environment),
                asked=asked,
                rule=settings.stuck,
                thinking=settings.thinking,
                instructions=self.instructions,
                prompt=self.prompt,
            )
            if message is None:
                continue
            asked.append(message.text)
            last_said = clock.now()
            if message.kind == "tell":
                await environment.say(message.text)
                continue
            question = message.text
            answer = await environment.ask(question)
            if answer is None:
                channel.add_context(AddedContext(f'Asked the person: "{question}" No answer came.'))
                continue
            facts = await _read_answer(step.facts_schema, self.answer_reader, question, answer)
            channel.add_context(AddedContext(f'Asked the person: "{question}" They answered: "{answer}"', facts))


async def _read_answer(
    facts_schema: type[BaseModel] | None, reader: Role, question: str, answer: str
) -> BaseModel | None:
    """The answer read into ``facts_schema`` (``None`` without one, or when it states nothing readable)."""
    if facts_schema is None:
        return None
    prompt = (
        f'The assistant asked: "{question}" The person answered: "{answer}". Fill in only what the answer states and '
        "leave everything else null; 'I don't know' states nothing."
    )
    try:
        return await reader.ask(None, prompt, response_format=facts_schema)
    except (StructuredOutputError, TimeoutError):
        return None
