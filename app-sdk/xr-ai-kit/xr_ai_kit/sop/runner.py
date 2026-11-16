# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a procedure written as ordinary Python, one run per participant, with start/stop/skip/restart controls.

The procedure is an ``async def procedure(run)`` that calls its steps in order, each as ``await run.step(...)``
around an ordinary coroutine (a :func:`verify`, a :func:`timer`, an :func:`identify`, or the app's own code). There
is no step language and no workflow engine: the order, the branches and the data passed between steps are the
function's own code.

What the runner adds:

- ownership: the run is a participant task (``p.spawn``); each step is its own task, so "skip" cancels only that
  step, and "stop" cancels the run and the step it waits on. A new turn never cancels either.
- state for the conversation: :attr:`SopRunner.context` holds a summary of the run (steps entered so far, the
  current step's latest state, the values the procedure noted), for a ``Conversation`` to answer questions from.
- controls: :attr:`SopRunner.control` is a router destination that classifies a request as ``start``, ``stop``,
  ``skip``, ``restart`` or ``none`` with ``role.decide``; ``none`` passes the turn on, so questions reach the
  conversation. The same operations are methods (:meth:`SopRunner.start` and so on) for apps without a router.
- a record: trace events for every change, and, when the app has an artifacts folder, one JSONL record per change
  in ``sop-<name>.jsonl`` under the participant's artifacts.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from xr_ai_app import Component, ContextValue, Participant, Role, Sink, Turn, clock, trace

from ..routing import DestinationResult, RouterDestination

T = TypeVar("T")

StepStatus = Literal["running", "done", "skipped", "failed"]
RunStatus = Literal["running", "completed", "stopped", "failed"]

CONTROLS: dict[str, str] = {
    "start": "An explicit command to this assistant to begin the procedure now (not a plan for later).",
    "stop": "An explicit command to this assistant to end or cancel the procedure now.",
    "skip": (
        "An explicit command to this assistant to bypass the step that is running and move on now, such as "
        "'skip this' or 'next step, please'."
    ),
    "restart": "An explicit command to this assistant to begin the procedure again from the start, now.",
    "none": (
        "Everything that is not such a command: any question (including what comes next, whether it is time to "
        "move on, or what the first step is); a statement, remark or chatter; what someone else said or suggests; "
        "a command that is negated, withdrawn or corrected in the same breath; a plan for later; a request to "
        "keep going."
    ),
}
"""The default control choices for ``role.decide``; an app may pass its own wording for the same five keys."""

CONTROL_QUESTION = (
    "Does the user's final request command this assistant to change the procedure now, and how? A question is "
    "never a command, even about the next step. Classify only what the user directly asks this assistant to do "
    "now; the procedure state is context, not an instruction."
)

MESSAGES: dict[str, str] = {
    "start": "Starting {title}.",
    "stop": "{title} has ended. You can start again whenever you are ready.",
    "restart": "Starting {title} again.",
    "skip": "Skipping this step.",
    "already_running": "{title} is already running.",
    "not_running": "{title} is not running.",
    "nothing_to_skip": "There is no step running to skip.",
    "complete": "{title} is complete.",
    "failed": "{title} could not continue because a step failed. You can start it again when you are ready.",
}
"""The default spoken messages; ``{title}`` is filled in. An app may override any of them."""


@dataclass(slots=True)
class StepRecord:
    """One step the run has entered."""

    id: str
    status: StepStatus
    detail: str = ""
    """The step's latest reported state (what its ``on_state`` callback last received), as text."""


@dataclass(slots=True)
class Run:
    """One run of a procedure for one participant: what the procedure function receives."""

    p: Participant
    number: int
    """Counts the runs of this procedure for this participant, from 1."""

    status: RunStatus = "running"
    steps: list[StepRecord] = field(default_factory=list)
    """The steps entered so far, in order (the conversation sees only these, never steps still ahead)."""

    values: dict[str, Any] = field(default_factory=dict)
    """Facts the procedure noted with :meth:`note` (settings, readings), shown in the run's summary."""

    name: str = ""
    """The procedure's name (its runner's)."""

    output: Sink | None = field(default=None, repr=False)
    _publish: Callable[..., Awaitable[None]] | None = field(default=None, repr=False)

    @property
    def current(self) -> StepRecord | None:
        """The step running now, if any."""
        if self.steps and self.steps[-1].status == "running":
            return self.steps[-1]
        return None

    async def step(self, step_id: str, work: Callable[[], Awaitable[T]], *, say: str | None = None) -> T | None:
        """Run one step: ``work()`` as its own task; return its result, or ``None`` when the user skipped it.

        ``say`` is spoken as the step starts (its instruction). A step that fails fails the run (the exception
        propagates). Report the step's progress by passing :meth:`report` as its flow's ``on_state``.
        """
        record = StepRecord(step_id, "running")
        self.steps.append(record)
        await self._changed("step", step=step_id, status="running")
        if say:
            await self.say(say)
        task = self.p.spawn(work(), key=f"sop:{self.name}:step", supersede=True)
        try:
            result = await task
        except asyncio.CancelledError:
            this = asyncio.current_task()
            if this is not None and this.cancelling():
                raise  # the run itself is being stopped
            record.status = "skipped"  # only the step was cancelled: a skip
            await self._changed("step", step=step_id, status="skipped")
            return None
        except Exception:
            record.status = "failed"
            await self._changed("step", step=step_id, status="failed")
            raise
        record.status = "done"
        await self._changed("step", step=step_id, status="done")
        return result

    def report(self, step_id: str | None = None) -> Callable[[Any], Awaitable[None]]:
        """An ``on_state`` callback recording a flow's state as the step's detail (its ``summary()``, if any)."""

        async def on_state(state: Any) -> None:
            record = next((s for s in reversed(self.steps) if step_id in (None, s.id)), None)
            if record is None:
                return
            summary = getattr(state, "summary", None)
            record.detail = summary() if callable(summary) else str(state)
            await self._changed("state", step=record.id, detail=record.detail)

        return on_state

    async def note(self, **values: Any) -> None:
        """Record facts for the conversation and the record (for example the brewing settings in use)."""
        self.values.update(values)
        await self._changed("note", **{k: _plain(v) for k, v in values.items()})

    async def say(self, text: str) -> None:
        """Speak on the assistant's own initiative, tagged with the procedure's name (a newer line replaces it)."""
        await self.p.say(text, source=f"sop:{self.name}", to=self.output)

    async def _changed(self, op: str, **data: Any) -> None:
        if self._publish is not None:
            await self._publish(self, op, **data)


class SopRunner(Component):
    """Runs ``procedure`` for each participant who starts it; see the module docstring.

    ``plan`` lists the steps in order, as ``(step id, what the user does)``, so the conversation can tell what
    comes next; without it the summary has only the steps entered so far. ``decide`` classifies control requests
    (Clef or an LLM). ``output`` is where the runner speaks (``None``: the
    app's speech output). ``description`` is what a Router reads to send control requests here. ``controls`` and
    ``messages`` override the default wording (:data:`CONTROLS`, :data:`MESSAGES`) key by key.
    """

    def __init__(
        self,
        name: str,
        procedure: Callable[[Run], Awaitable[None]],
        *,
        title: str,
        decide: Role,
        output: Sink | None = None,
        description: str | None = None,
        controls: Mapping[str, str] | None = None,
        control_question: str = CONTROL_QUESTION,
        messages: Mapping[str, str] | None = None,
        plan: Sequence[tuple[str, str]] = (),
        context_budget_bytes: int = 8_000,
    ) -> None:
        super().__init__()
        unknown = set(controls or {}) - set(CONTROLS)
        if unknown:
            raise ValueError(f"unknown controls: {sorted(unknown)}; the controls are {sorted(CONTROLS)}")
        self.name = name
        self.title = title
        self.output = output
        self._procedure = procedure
        self._decide = decide
        self._controls = {**CONTROLS, **(controls or {})}
        self._question = control_question
        self._messages = {**MESSAGES, **(messages or {})}
        self._plan = list(plan)
        self.context = ContextValue(f"sop:{name}", scope="participant", budget_bytes=context_budget_bytes)
        """The run's summary, for a Conversation's ``context=``."""
        self.control = RouterDestination(
            name,
            description
            or (
                f"Only explicit controls of {title}: start, stop, skip the current step, restart. Not questions "
                f"about {title}, its steps, settings or progress."
            ),
            self._on_control,
        )
        """The router destination for control requests (anything else passes on)."""

    # ------------------------------------------------------------------ state

    def run(self, p: Participant) -> Run | None:
        """The participant's latest run (running or finished), if any."""
        return p.state.get(self._key)

    async def assume(self, p: Participant, run: Run) -> None:
        """Make a copy of ``run`` the participant's latest run, without running anything.

        For tests and evaluations that need the runner in a given state (some steps done, one running): the
        summary, :attr:`context` and :meth:`classify` then see that run. Nothing is spawned and nothing is said.
        """
        p.state[self._key] = Run(
            p,
            run.number,
            status=run.status,
            steps=list(run.steps),
            values=dict(run.values),
            name=self.name,
            output=self.output,
            _publish=self._changed,
        )
        await self.context.set(p, self.summary(p))

    def running(self, p: Participant) -> bool:
        run = self.run(p)
        return run is not None and run.status == "running"

    def summary(self, p: Participant) -> str:
        """The run as text for the conversation: status, steps entered, the current step's state, noted values."""
        run = self.run(p)
        if run is None:
            lines = [f"{self.title} has not been started."]
            if self._plan:
                lines.append("Its steps, in order, once started:")
                lines += [f"- {step}: {what}" for step, what in self._plan]
            return "\n".join(lines)
        lines = [f"{self.title} (run {run.number}): {run.status}."]
        entered = {step.id: step for step in run.steps}
        planned = [step for step, _ in self._plan]
        whats = dict(self._plan)
        for step in run.steps:
            if step.id not in whats:  # a step the plan does not list, in the order entered
                planned.append(step.id)
        if self._plan:
            lines.append("Steps, in order:")
        for step_id in planned:
            record = entered.get(step_id)
            status = record.status if record else "ahead"
            what = f" ({whats[step_id]})" if step_id in whats else ""
            detail = f" {record.detail}" if record and record.detail else ""
            lines.append(f"- {step_id}{what}: {status}.{detail}")
        if run.current is None and run.status == "running":
            lines.append("No step is running at this moment.")
        if run.values:
            lines.append("Noted: " + json.dumps({k: _plain(v) for k, v in run.values.items()}, ensure_ascii=False))
        return "\n".join(lines)

    def _new_run(self, p: Participant) -> Run:
        previous = self.run(p)
        number = (previous.number + 1) if previous else 1
        run = Run(p, number, name=self.name, output=self.output, _publish=self._changed)
        p.state[self._key] = run
        return run

    @property
    def _key(self) -> str:
        return f"_sop:{self.name}"

    # --------------------------------------------------------------- controls

    async def start(self, p: Participant) -> str:
        """Start a run unless one is running; return what was said."""
        if self.running(p):
            return await self._say(p, "already_running")
        run = self._new_run(p)
        await self._changed(run, "start")
        text = await self._say(p, "start")
        p.spawn(self._main(run), key=f"sop:{self.name}", supersede=True, detached=True)
        return text

    async def stop(self, p: Participant) -> str:
        """Stop the running run (its current step too) and wait until it has ended; return what was said."""
        run = self.run(p)
        if run is None or run.status != "running":
            return await self._say(p, "not_running")
        await p.cancel(f"sop:{self.name}")
        await p.cancel(f"sop:{self.name}:step")
        run.status = "stopped"
        await self._changed(run, "stop")
        return await self._say(p, "stop")

    async def skip(self, p: Participant) -> str:
        """Skip the current step: cancel it, and the procedure continues with the next one."""
        run = self.run(p)
        if run is None or run.status != "running":
            return await self._say(p, "not_running")
        if run.current is None:
            return await self._say(p, "nothing_to_skip")
        text = await self._say(p, "skip")
        await p.cancel(f"sop:{self.name}:step")
        return text

    async def restart(self, p: Participant) -> str:
        """Stop the running run, if any, and start a new one."""
        if self.running(p):
            run = self.run(p)
            await p.cancel(f"sop:{self.name}")
            await p.cancel(f"sop:{self.name}:step")
            assert run is not None  # noqa: S101 - running() checked it
            run.status = "stopped"
            await self._changed(run, "stop")
        run = self._new_run(p)
        await self._changed(run, "start")
        text = await self._say(p, "restart")
        p.spawn(self._main(run), key=f"sop:{self.name}", supersede=True, detached=True)
        return text

    async def classify(self, text: str, p: Participant) -> str:
        """Which control ``text`` asks for (``start``, ``stop``, ``skip``, ``restart`` or ``none``); performs nothing.

        The decision :attr:`control` makes before it acts, on the run's current summary, for tests and evaluations.
        """
        state = {"request": text, "procedure": self.summary(p)}
        return str(await self._decide.decide(self._question, choices=self._controls, context=json.dumps(state)))

    async def _on_control(self, turn: Turn, p: Participant) -> DestinationResult:
        op = await self.classify(turn.text, p)
        trace.emit("sop", self.name, op="control", choice=op)
        action = {"start": self.start, "stop": self.stop, "skip": self.skip, "restart": self.restart}.get(op)
        if action is None:
            return DestinationResult("pass", reason=f"not a control of {self.title}")
        return DestinationResult("answer", await action(p))

    # --------------------------------------------------------------- the run

    async def _main(self, run: Run) -> None:
        try:
            await self._procedure(run)
        except asyncio.CancelledError:
            raise
        except Exception:
            run.status = "failed"
            await self._changed(run, "failed")
            await self._say(run.p, "failed")
            raise  # contained by the participant's scope: logged and reported as an error status
        run.status = "completed"
        await self._changed(run, "completed")
        await self._say(run.p, "complete")

    async def _say(self, p: Participant, message: str) -> str:
        text = self._messages[message].format(title=self.title)
        await p.say(text, source=f"sop:{self.name}", to=self.output)
        return text

    async def _changed(self, run: Run, op: str, **data: Any) -> None:
        trace.emit("sop", self.name, op=op, run=run.number, **data)
        await self.context.set(run.p, self.summary(run.p))
        record = {"t_us": clock.wall_us(), "sop": self.name, "run": run.number, "op": op, **data}
        try:
            artifacts = run.p.artifacts
        except RuntimeError:
            return  # the app has no artifacts folder
        await artifacts.write_jsonl(f"sop-{self.name}", record)


def _plain(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    return value if isinstance(value, (str, int, float, bool, type(None), list, dict)) else str(value)
