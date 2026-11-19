# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kit's escalator (:class:`ReasoningEscalator`): an agent of its own, beside one step, that helps the person
when the step cannot decide, by reasoning in text over the step's evidence. (Any :class:`~.escalation_channel.Escalator`
can stand in its place: one that sends the frame behind a piece of evidence to a model of its own, say.)

It knows the step only through the run's :class:`~.escalation_channel.EscalationChannel`: it reads the step's
snapshot (what it does, its recent evidence, what it knows) and adds context to it (what the person said). It talks
to the person itself, through the step's own environment (a :class:`~..steps.step_api.StepEnvironment`), never
through the step, and it
never completes, skips or slows the step. Which step it is beside is :class:`~.step_with_escalator.StepWithEscalator`.

While the step runs it looks at the step's evidence every ``every_s``; when most of the latest is undecided
(:func:`is_stuck`), a reasoning model reads it with the step's task, ``guidance`` (how the person can help) and what
is known (:func:`decide_escalation`) and infers what is missing and whether the person can help: by telling it (a
question, whose answer is added to the step's context) or by doing something, such as bringing the kettle back into
view (an instruction, spoken only). It sees no frames: the step already looked.

What the person says comes first. With ``learns_from_conversation``, it reads the conversation
(``environment.conversation``) into the step's facts as the step starts and again each time it changes
(:func:`read_facts_from_conversation`), stuck or not: a unit said when the target was announced, or volunteered while
the kettle heats, reaches the step without anyone asking, and the step decides with it before the escalator says
anything.

Its prompt is per step: its words (:class:`EscalationWords`) give the guidance and may replace the closing
``instructions`` or the whole ``prompt`` (:data:`PROMPT`, a template).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from xr_ai_app import Role, StructuredOutputError, clock, trace

from ..steps.step_api import StepEnvironment
from .escalation_channel import AddedContext, EscalationChannel, Evidence, StepSnapshot


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


def is_stuck(evidence: Sequence[Evidence], *, undecided: int = 3, of_last: int = 4) -> bool:
    """Whether the step has mostly stopped deciding: at least ``undecided`` of its last ``of_last`` pieces of evidence
    are undecided.

    One undecided piece among decisions is noise; mostly undecided ones, even with a stray misread among them (a
    number read off a display cut off by the edge of the view), mean the step is missing something.
    """
    recent = evidence[-of_last:]
    return len(recent) >= of_last and sum(not e.decided for e in recent) >= undecided


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
    words: EscalationWords,
    settings: EscalationSettings | None = None,
    *,
    situation: str = "",
    said_lines: Sequence[str] = (),
) -> EscalationMessage | None:
    """One judgement over a snapshot of the step: what to say to the person now, or ``None``.

    ``reason`` is a reasoning text model. It reads ``words.prompt`` filled in: the step's task, ``words.guidance``
    (how the person can help), the step's recent evidence, ``situation`` (what is going on), what the step knows, the
    lines already said to the person (``said_lines``) and ``words.instructions``. ``None`` without guidance, while the
    step is not :func:`is_stuck`, on an unreadable reply, or for a line already said. Asks nobody: an evaluation calls
    it on recorded evidence.
    """
    settings = settings or EscalationSettings()
    stuck = is_stuck(step.evidence, undecided=settings.stuck_undecided, of_last=settings.stuck_of_last)
    if not words.guidance or not stuck:
        return None
    text = words.prompt.format(
        task="\n".join(step.task),
        guidance=words.guidance,
        evidence="\n".join(f"- {e.label}: {e.description}" for e in step.evidence),
        situation=f"\nWhat is going on:\n{situation}" if situation else "",
        known=(
            f"Already known:\n{step.knowledge_text}"
            if step.knowledge_text
            else "Nothing has been learned from the person yet."
        ),
        asked=("\nAlready said to the person: " + " | ".join(said_lines)) if said_lines else "",
        instructions=words.instructions,
    )
    try:
        decision = await reason.ask(None, text, response_format=_Decision, thinking=settings.thinking)
    except (StructuredOutputError, TimeoutError):
        return None
    said = (decision.words or "").strip()
    if decision.action == "none" or not said or said in said_lines:
        return None
    return EscalationMessage(decision.action, said)


CONVERSATION_READING = (
    "The conversation so far (user: the person; assistant: the assistant):\n{conversation}\n\n"
    "Fill in only what the person has stated about their own situation, and leave everything else null. The "
    "assistant's words state nothing; neither do the person's questions, guesses, what-ifs or talk of other things."
)
"""How the conversation is read into a step's facts (``{conversation}`` is filled in)."""


async def read_facts_from_conversation(
    facts_schema: type[BaseModel], reader: Role, conversation: str
) -> BaseModel | None:
    """The facts the person stated in ``conversation`` (its text so far), read into ``facts_schema``: read twice
    at once, and only the facts both reads agree on; ``None`` when either reply is unreadable. Asks nobody: an
    evaluation calls it on a recorded conversation.

    A conversation is read again on every change, and a fact once given stays, so a rare stray fill (the assistant's
    "80 degrees Celsius" taken as the person's unit) would add up over a step; two reads agreeing make it rarer
    still, at no cost in time.
    """
    prompt = CONVERSATION_READING.format(conversation=conversation)
    try:
        first, second = await asyncio.gather(
            reader.ask(None, prompt, response_format=facts_schema),
            reader.ask(None, prompt, response_format=facts_schema),
        )
    except (StructuredOutputError, TimeoutError):
        return None
    other = second.model_dump(mode="json", exclude_none=True)
    agreed = {
        name: value
        for name, value in first.model_dump(mode="json", exclude_none=True).items()
        if other.get(name) == value
    }
    try:
        return facts_schema.model_validate(agreed)
    except ValidationError:  # a field without a default that the reads did not agree on
        return None


class EscalationWords(BaseModel):
    """An escalator's words for one step (a step folder's ``escalation.yaml``): its ``guidance``, and optionally this
    step's own closing ``instructions`` or whole ``prompt``."""

    model_config = ConfigDict(extra="forbid")
    guidance: str
    """How the person can help when the step cannot decide."""
    instructions: str = INSTRUCTIONS
    """What the reasoning model is asked to do (default :data:`INSTRUCTIONS`)."""
    prompt: str = PROMPT
    """The whole prompt (default :data:`PROMPT`)."""


class EscalationSettings(BaseModel):
    """How one escalator works (an app's settings may embed this, per step or for all)."""

    model_config = ConfigDict(extra="forbid")
    reads_run_summary: bool = False
    """Whether it reads the run's summary (the plan, the steps' states, noted values) when it decides what to say."""
    reads_conversation: bool = False
    """Whether it reads the conversation with the person when it decides what to say (the person may already have
    said what it would ask)."""
    learns_from_conversation: bool = False
    """Whether it reads the conversation into the step's facts as the step starts and on each change: what the person
    says reaches the step without anyone asking. Only for a step that accepts facts (a ``facts_schema``)."""
    every_s: float = 20.0
    """How often it looks at the step's evidence."""
    max_lines: int = 2
    """How many things it may say during one step."""
    min_gap_s: float = 30.0
    """At least this long between two things it says."""
    stuck_undecided: int = 3
    stuck_of_last: int = 4
    """The step is stuck when at least ``stuck_undecided`` of its last ``stuck_of_last`` pieces of evidence are
    undecided (:func:`is_stuck`)."""
    thinking: bool | None = True
    """Whether the reasoning model thinks before it decides (``None``: the role's default)."""


class ReasoningEscalator:
    """The kit's escalator for one step; see the module docstring.

    Built from its ``words`` (:class:`EscalationWords`: ``guidance``, and this step's ``instructions`` and
    ``prompt``). ``reason`` reasons over the step's evidence; ``facts_reader`` (a text model; default ``reason``)
    reads the person's answers, and the conversation, into the typed facts the step accepts.
    """

    scoring: ClassVar[tuple[str, ...]] = (f"{__package__}.escalation_scoring",)
    """What an evaluation case may expect of it (``asks``, ``says``, ``quiet``)."""

    def __init__(
        self,
        words: EscalationWords,
        reason: Role,
        settings: EscalationSettings | None = None,
        *,
        facts_reader: Role | None = None,
    ) -> None:
        self.words = words
        self.reason = reason
        self.facts_reader = facts_reader or reason
        self.settings = settings or EscalationSettings()

    async def run(self, channel: EscalationChannel, environment: StepEnvironment) -> None:
        """Run until cancelled, two loops side by side once the step has started:

        - :meth:`_follow_conversation`, with ``learns_from_conversation``: read the conversation into the step's facts
          at once, and again each time it changes;
        - :meth:`_help_when_stuck`: every ``every_s`` seconds, if the step is stuck, decide what to say, and pass
          answers on. While the conversation is being read, or within ``every_s`` of the step being given facts, it
          says nothing: the step decides with them first.

        An error in one turn of either loop is traced and that loop goes on; neither ends the other.
        """
        await channel.step_started()
        learning = _Learning()
        async with asyncio.TaskGroup() as both:
            both.create_task(self._follow_conversation(channel, environment, learning))
            both.create_task(self._help_when_stuck(channel, environment, learning))

    async def _describe_situation(self, step: StepSnapshot, environment: StepEnvironment) -> str:
        """What it reads about what is going on: how long the step has run, then the run's summary and the
        conversation, as its settings say."""
        lines = [f"The step '{step.step_id}' has been running for {clock.now() - step.started_s:.0f} s."]
        if self.settings.reads_run_summary and (summary := await environment.run_summary()):
            lines.append(summary)
        if self.settings.reads_conversation and (said := await environment.conversation.text()):
            lines.append(f"The conversation so far:\n{said}")
        return "\n".join(lines)

    async def _help_when_stuck(
        self, channel: EscalationChannel, environment: StepEnvironment, learning: _Learning
    ) -> None:
        """Every ``every_s``: if the step is stuck, ask or tell the person what is missing. At most ``max_lines``
        lines, never the same one twice, at least ``min_gap_s`` apart. A question is asked with ``environment.ask``
        and its answer added to the step's context; an instruction is spoken with ``environment.say``."""
        settings = self.settings
        said_lines: list[str] = []
        last_said_s: float | None = None
        while len(said_lines) < settings.max_lines:
            await asyncio.sleep(settings.every_s)  # also what keeps a failing turn from spinning
            step = channel.snapshot()
            if step is None or not is_stuck(
                step.evidence, undecided=settings.stuck_undecided, of_last=settings.stuck_of_last
            ):
                continue  # deciding: nothing is missing
            if learning.reading_conversation:
                continue  # the step may know more in a moment: let it decide with that first
            if learning.last_given_s is not None and clock.now() - learning.last_given_s < settings.every_s:
                continue  # the step was just given facts: give it one look period to use them
            if last_said_s is not None and clock.now() - last_said_s < settings.min_gap_s:
                continue
            try:
                message = await decide_escalation(
                    step,
                    self.reason,
                    self.words,
                    settings,
                    situation=await self._describe_situation(step, environment),
                    said_lines=said_lines,
                )
                if message is None:
                    continue
                said_lines.append(message.text)
                last_said_s = clock.now()
                if message.kind == "tell":
                    await environment.say(message.text)
                    continue
                answer = await environment.ask(message.text)
                if answer is None:
                    channel.add_context(AddedContext(f'Asked the person: "{message.text}" No answer came.'))
                    continue
                facts = await _read_answer(step.facts_schema, self.facts_reader, message.text, answer)
                learning.give(facts)
                channel.add_context(
                    AddedContext(f'Asked the person: "{message.text}" They answered: "{answer}"', facts)
                )
            except Exception as exc:  # noqa: BLE001 - one failed turn (a model server error) never ends the escalator
                trace.emit("sop", step.step_id, op="escalator_error", loop="help_when_stuck", error=repr(exc))

    async def _follow_conversation(
        self, channel: EscalationChannel, environment: StepEnvironment, learning: _Learning
    ) -> None:
        """Read the conversation into the step's facts now and after each change; add only what the step has not
        been given. Nothing to follow without ``learns_from_conversation`` or a step that accepts facts."""
        step = channel.snapshot()
        if not self.settings.learns_from_conversation or step is None or step.facts_schema is None:
            return
        conversation = environment.conversation
        last_read = ""
        async with contextlib.aclosing(conversation.changes()) as changes:
            async for _ in changes:  # a failed turn waits for the next change: it never spins
                try:
                    said = await conversation.text()
                    if not said or said == last_read:
                        continue
                    last_read = said
                    learning.reading_conversation = True
                    try:
                        facts = await read_facts_from_conversation(step.facts_schema, self.facts_reader, said)
                    finally:
                        learning.reading_conversation = False
                    if stated := learning.new_in(facts):
                        learning.give(facts)
                        channel.add_context(AddedContext(f"From the conversation so far: {json.dumps(stated)}", facts))
                except Exception as exc:  # noqa: BLE001 - as in _help_when_stuck
                    trace.emit("sop", step.step_id, op="escalator_error", loop="follow_conversation", error=repr(exc))


@dataclass(slots=True)
class _Learning:
    """What one run of an escalator has given its step, shared by its two loops."""

    facts: dict[str, Any] = field(default_factory=dict)
    """The typed facts given so far, by field (from answers and from the conversation)."""
    last_given_s: float | None = None
    """When new facts were last given to the step (core clock)."""
    reading_conversation: bool = False
    """Whether the conversation is being read now: its facts may arrive any moment."""

    def new_in(self, facts: BaseModel | None) -> dict[str, Any]:
        """The facts ``facts`` states (all of them) if any of them is new; empty otherwise."""
        stated = facts.model_dump(mode="json", exclude_none=True) if facts is not None else {}
        return stated if any(self.facts.get(name) != value for name, value in stated.items()) else {}

    def give(self, facts: BaseModel | None) -> None:
        """Record that the step was given ``facts`` (when any of them is new)."""
        if stated := self.new_in(facts):
            self.facts.update(stated)
            self.last_given_s = clock.now()


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
