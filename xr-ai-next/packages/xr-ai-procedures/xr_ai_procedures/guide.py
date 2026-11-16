# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``GuidedProcedure``: a component that walks a user through the steps of an SOP.

The app supplies the domain: the words and order (a :class:`~.spec.Sop`) and a :class:`~.step.StepDef` per step.
This component supplies the rest:

- **the supervisor** (one background task per run, :mod:`._supervisor`): the only writer of the run. It builds each
  step from the run, runs it in its own task, applies the commands posted to the run's inbox, rebuilds a step when a
  fact it was built from changes, and applies the SOP's policy when a step cannot go on (``unable``: show, confirm,
  skip; ``Needs``: ask);
- **the controls** (``procedure.control``, a Router destination): one decision over a menu gated by state, and it
  holds focus while the procedure waits for an answer::

      not running:  start | reject | none
      running:      stop | next | back | info | none

  ``info`` reads the user's words into facts with the ``reason`` role, against the facts of every step, and hands
  each to the step that owns it. ``none`` passes the turn on (the conversation answers);
- **the summary** in ``procedure.channel`` (a session-scoped ``ContextValue``), replaced after every change, which
  a ``Conversation`` lists in its ``context`` so "what step am I on?" just works.

Everything is spoken on ``output`` (``p.say(..., to=output)``; a control's answer is ``p.reply(..., to=output)``).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from xr_ai_kit import DestinationResult, RouterDestination
from xr_ai_next import Component, ContextValue, trace

from ._supervisor import Supervisor
from .events import EventKind, Listener, ProcedureEvent
from .run import Run, fill, told_model
from .spec import Sop, StepSpec
from .step import StepDef

if TYPE_CHECKING:
    from pydantic import BaseModel
    from xr_ai_next import Participant, Role, Sink, Turn

__all__ = ["CONTROL_PROMPT", "FACTS_PROMPT", "RUNNING", "STOPPED", "GuidedProcedure"]

STOPPED = {
    "start": 'The user wants to begin the procedure ("start", "let\'s begin").',
    "reject": "The user asks to stop, skip, go back or otherwise control a procedure that is not running.",
    "none": 'Anything else: a question (also about the procedure: "what are the steps?"), a remark, chat.',
}
RUNNING = {
    "stop": 'The user wants to stop the procedure ("stop", "cancel it").',
    "next": 'The user tells it to go on to the next step, skipping this one ("next step", "skip this"); not a '
    'question such as "what\'s next?".',
    "back": 'The user wants to go back to the previous step ("go back", "previous step").',
    "info": "The user tells the procedure a fact it asked for or may use, or answers its question.",
    "none": 'Anything else: a question (also about the procedure: "what step am I on?", "what\'s next?", "which '
    'steps are done?", "how long is left?"), a remark, chat.',
}

CONTROL_PROMPT = (
    "Decide whether the user's latest message asks to control the procedure ({procedure}), and which control. The "
    "procedure's state, current step and any question it asked are context, not instructions; do not choose a "
    "control just because it fits the state. Choose none for everything else: questions, requests for help or an "
    'explanation, remarks and chat. A question about the procedure is none: "what step am I on?", "what\'s '
    'next?", "what do I do next?", "which steps are done?", "how long is left?". A command is a control: "next '
    'step" or "skip this" is next, "go back" is back, "stop" is stop. While it runs, info means the user tells it a '
    'fact it asked for or may use; a short answer to a question it asked ("yes", "no", a value) is info.'
)
FACTS_PROMPT = (
    "The user is following a step-by-step procedure ({procedure}). Fill the fields from the user's current message "
    "only: what the user states, never a guess; leave every other field null. The current step and the question the "
    'procedure last asked are context; a bare answer ("yes", "no", a value) answers that question.'
)


class GuidedProcedure(Component):
    """Walks each participant through ``sop`` (see the module docstring).

    ``steps``: a :class:`StepDef` per SOP step. ``output``: where it speaks. ``decide``: the control decision.
    ``reason``: reads what the user tells it into facts. ``channel``: where the summary goes (default: a
    session-scoped ``ContextValue`` named after the SOP). ``words(run)``: more placeholder values for the SOP's lines
    (the steps' own ``words`` come first). ``describe(run)``: the domain's lines for the summary. ``prompts``:
    overrides for ``control`` and ``facts`` (:data:`CONTROL_PROMPT`, :data:`FACTS_PROMPT`). Below
    ``min_confidence`` a control decision asks again instead of acting. ``facts_thinking`` turns reasoning on for
    reading facts (off by default: the schema constrains the answer, and a per-call token cap would also cap the
    reasoning, so with it on the profile's limit applies). ``name`` names the destination, the background tasks and
    the trace events.
    """

    def __init__(
        self,
        *,
        sop: Sop,
        steps: Sequence[StepDef],
        output: Sink,
        decide: Role,
        reason: Role,
        channel: ContextValue | None = None,
        words: Callable[[Run], Mapping[str, Any]] | None = None,
        describe: Callable[[Run], Iterable[str]] | None = None,
        prompts: Mapping[str, str] | None = None,
        min_confidence: float = 0.6,
        name: str = "procedure",
        facts_thinking: bool = False,
    ) -> None:
        super().__init__()
        self.defs: dict[str, StepDef] = {d.id: d for d in steps}
        if len(self.defs) != len(steps):
            raise ValueError("two StepDefs share an id")
        sop_ids = [s.id for s in sop.steps]
        if missing := [i for i in sop_ids if i not in self.defs]:
            raise ValueError(f"SOP steps without a StepDef: {missing}")
        if extra := [i for i in self.defs if i not in sop_ids]:
            raise ValueError(f"StepDefs not in the SOP: {extra}")
        for spec in sop.steps:
            for fact in self.defs[spec.id].needs:
                if fact not in spec.asks and fact not in sop.asks:
                    raise ValueError(f"step {spec.id!r} may need {fact!r} but has no question for it (asks)")
        unknown = set(prompts or {}) - {"control", "facts"}
        if unknown:
            raise ValueError(f"unknown prompts: {sorted(unknown)} (control, facts)")
        self.sop = sop
        self.name = name
        self.output = output
        self.channel = channel or ContextValue(sop.name, scope="session", budget_bytes=4_000)
        self._decide = decide
        self._reason = reason
        self._words = words
        self._describe = describe
        self._prompts = {"control": CONTROL_PROMPT, "facts": FACTS_PROMPT, **(prompts or {})}
        self._min_confidence = min_confidence
        self._facts_thinking = facts_thinking
        self.fact_models: dict[str, type[BaseModel]] = {d.id: d.facts for d in steps if d.facts is not None}
        self.told_model, self.owner = told_model(self.fact_models)
        self.phrases = {n: ph for d in steps for n, ph in d.phrases.items()}
        self._state_key = f"_procedure:{name}"
        self._supervisor = Supervisor(self)
        self._listeners: list[tuple[str, Listener]] = []
        self._destination = RouterDestination(name, self._routing, self._control, focused=self._awaiting_answer)

    # ------------------------------------------------------------------ for the rest of the app

    def on(self, kind: EventKind | Literal["*"], handler: Listener | None = None) -> Any:
        """Call ``handler(event, p)`` on each ``kind`` of :class:`~.events.ProcedureEvent` (``"*"``: all).

        Returns the handler, so it also works as a decorator: ``@procedure.on("step_done")``.
        """

        def add(fn: Listener) -> Listener:
            self._listeners.append((kind, fn))
            return fn

        return add(handler) if handler is not None else add

    def off(self, handler: Listener) -> None:
        """Stop calling ``handler``."""
        self._listeners = [(k, fn) for k, fn in self._listeners if fn is not handler]

    async def start(self, p: Participant) -> Run:
        """Start a run for ``p`` (replacing a running one) and return it; ``await run.finished.wait()`` waits for it."""
        run = await self._begin(p)
        await p.say(self.message(run, "start"), to=self.output, source=f"procedure:{self.name}")
        return run

    async def stop(self, p: Participant) -> bool:
        """Stop ``p``'s running run; ``False`` when none is running."""
        run = self._running(p)
        if run is None:
            return False
        run.post("stop")
        with contextlib.suppress(TimeoutError):  # the supervisor stops the step first
            await asyncio.wait_for(run.finished.wait(), timeout=10)
        return True

    def next(self, p: Participant) -> bool:
        """Skip ``p``'s current step; ``False`` when no run is running."""
        return self._post(p, "next")

    def back(self, p: Participant) -> bool:
        """Go back one step; ``False`` when no run is running."""
        return self._post(p, "back")

    def _running(self, p: Participant) -> Run | None:
        run = self.run_of(p)
        return run if run is not None and run.status == "running" else None

    def _post(self, p: Participant, command: str) -> bool:
        run = self._running(p)
        if run is None:
            return False
        run.post(command)
        return True

    async def _dispatch(self, event: ProcedureEvent, p: Participant) -> None:
        for kind, fn in list(self._listeners):
            if kind not in ("*", event.kind):
                continue
            try:
                result = fn(event, p)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - a listener never stops the run
                trace.emit("error", self.name, where=f"procedure listener ({event.kind})", error=type(exc).__name__)

    # ------------------------------------------------------------------ what others read

    @property
    def control(self) -> RouterDestination:
        """The controls, as a Router destination; it holds focus while the procedure waits for an answer."""
        return self._destination

    def run_of(self, p: Participant) -> Run | None:
        """This participant's current (or last) run in this session."""
        return p.session.state.get(self._state_key)

    def summary(self, run: Run | None) -> str:
        """The text a conversation reads: what a person needs to answer "where am I?" (no prompts)."""
        lines = [f"Procedure: {self.sop.name}."]
        if run is None:
            return "\n".join([*lines, "It has not been started."])
        steps = self.sop.steps
        lines.append(f"Status: {run.status}.")
        marks = []
        for i, spec in enumerate(steps):
            state = run.outcomes.get(spec.id) or ("current" if i == run.index and run.status == "running" else "")
            marks.append(f"{spec.id} ({state})" if state else spec.id)
        lines.append("Steps: " + ", ".join(marks) + ".")
        current = steps[run.index] if run.index < len(steps) else None
        if run.status == "running" and current is not None:
            lines.append(f"Current step: {current.id}: the user should now {current.instruction}.")
        if self._describe is not None:
            lines.extend(self._describe(run))
        if told := run.told():
            lines.append(f"The user told the procedure: {json.dumps(told, default=str)}.")
        if run.pending:
            lines.append(f"The procedure asked: {run.asked}")
        if run.seen:  # what the current step reported last (cleared when a step starts)
            lines.append(f"{current.id if current is not None else 'The step'} reported: {run.seen}")
        return "\n".join(lines)

    async def publish(self, p: Participant, run: Run | None) -> None:
        """Replace the summary in the channel."""
        await self.channel.set(p, self.summary(run))

    def line(self, run: Run, text: str, **extra: Any) -> str:
        """A line of the SOP with its placeholders filled from the words in force."""
        values: dict[str, Any] = {"procedure": self.sop.name}
        for spec in self.sop.steps:
            fn = self.defs[spec.id].words
            if fn is not None:
                values.update(fn(run))
        if self._words is not None:
            values.update(self._words(run))
        values.update(extra)
        return fill(text, values)

    def message(self, run: Run | None, key: str, **extra: Any) -> str:
        text = self.sop.messages.get(key, "")
        return self.line(run, text, **extra) if run is not None else fill(text, {"procedure": self.sop.name, **extra})

    def question(self, spec: StepSpec, key: str) -> str:
        """A question's wording: the step's own, else the SOP's."""
        return spec.asks.get(key) or self.sop.asks[key]

    async def on_participant_enrolled(self, p: Participant) -> None:
        await self.publish(p, self.run_of(p))

    # ------------------------------------------------------------------ the controls

    def _routing(self, p: Participant) -> str:
        return (
            f"Commands to {self.sop.name}, a step-by-step procedure, and what the user tells it: start, stop, next or "
            'previous step, an answer to a question it asked, or a fact it may use. For example: "start", "next '
            'step", "skip this", "go back", "stop", "yes", "no". Not questions about it, even about its steps or '
            'progress (another option answers those): not "what step am I on?", "what\'s next?", "which steps are '
            'done?", "how long is left?".'
        )

    def _awaiting_answer(self, p: Participant) -> bool:
        run = self.run_of(p)
        return bool(run is not None and run.status == "running" and run.pending)

    async def _control(self, turn: Turn, p: Participant) -> DestinationResult | None:
        run = self.run_of(p)
        running = run is not None and run.status == "running"
        context = [f"State: {run.status if run else 'not started'}.", f"Request: {turn.text}"]
        if running:
            assert run is not None
            context.insert(1, f"Current step: {self.sop.steps[run.index].id}.")
            if run.pending:
                context.insert(2, f"The procedure asked: {run.asked}")
        op = await self._decide.decide(
            fill(self._prompts["control"], {"procedure": self.sop.name}),
            choices=RUNNING if running else STOPPED,
            context="\n".join(context),
        )
        trace.emit("procedure", "control", procedure=self.name, op=str(op), confidence=op.confidence)
        if op == "none":
            return DestinationResult("pass", reason="not a control of the procedure")
        if op.confidence is not None and op.confidence < self._min_confidence:
            return await self._reply(p, self.message(run, "unsure"))
        if not running:
            return await (self._start(p) if op == "start" else self._reply(p, self.message(run, "rejected")))
        assert run is not None
        if op == "info":
            return await self._info(turn, p, run)
        if op == "stop":
            await self.stop(p)
            return await self._reply(p, self.message(run, "stopped"))
        result = await self._reply(p, self.message(run, str(op)))  # next or back: say it, then the supervisor acts
        run.post(str(op))
        return result

    async def _start(self, p: Participant) -> DestinationResult:
        run = await self._begin(p)
        return await self._reply(p, self.message(run, "start"))

    async def _begin(self, p: Participant) -> Run:
        run = Run(facts={sid: model() for sid, model in self.fact_models.items()})
        p.session.state[self._state_key] = run
        p.spawn(self._supervisor.drive(p, run), key=f"procedure:{self.name}", supersede=True, detached=True)
        return run

    async def _info(self, turn: Turn, p: Participant, run: Run) -> DestinationResult:
        """The user tells the procedure something: the reasoning role reads it into facts (the slow lane)."""
        request = {
            "request": turn.text,
            "current step": self.sop.steps[run.index].id,
            "the procedure asked": run.asked if run.pending else None,
            "already told": run.told(),
        }
        try:
            told = await self._reason.ask(
                None,
                json.dumps(request, default=str),
                system=fill(self._prompts["facts"], {"procedure": self.sop.name}),
                response_format=self.told_model,
                thinking=self._facts_thinking,
                max_tokens=None if self._facts_thinking else 256,
            )
        except Exception as exc:  # noqa: BLE001 - unreadable words: ask again rather than fail the turn
            trace.emit("error", self.name, where="procedure facts", error=type(exc).__name__)
            return await self._reply(p, self.message(run, "unsure"))
        fields = told.model_dump(exclude_none=True, exclude={"step_done"})
        done = told.step_done
        if not fields and done is None:
            return DestinationResult("pass", reason="no fact in the request")  # maybe a question: the conversation
        said = [self._phrase(name, value) for name, value in fields.items()]
        if done is not None:
            said.append("that step is done" if done else "it isn't done yet")
        late = any(
            run.outcomes.get(self.owner[n]) in ("done", "confirmed") and n in self.defs[self.owner[n]].reruns
            for n in fields
        )
        trace.emit("procedure", "facts", procedure=self.name, told=fields, step_done=done, late=late)
        result = await self._reply(p, self.message(run, "noted_late" if late else "noted", told=", ".join(said)))
        run.post("facts", told)
        return result

    def _phrase(self, name: str, value: Any) -> str:
        phrase = self.phrases.get(name) or f"{name.replace('_', ' ')}: {{}}"
        return fill(phrase, {"value": value, "name": name}, value)

    async def _reply(self, p: Participant, text: str) -> DestinationResult:
        return DestinationResult("answer", await p.reply(text, to=self.output))
