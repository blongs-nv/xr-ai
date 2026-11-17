# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a procedure written as ordinary Python, one run per participant, with start/stop/skip/restart controls.

The procedure is an ``async def procedure(run)`` that runs its steps in order, each as ``await run.step(step)``
(a :mod:`..steps` object: a watch, a read, a wait) or ``await run.step(step_id, work)`` around any coroutine of the
app's own. There is no step language and no workflow engine: the order, the branches and the data passed between
steps are the function's own code. :class:`~.run.Run` is what a step touches (it is a
:class:`~..steps.step_api.StepIO`).

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
from typing import Any

from xr_ai_app import Component, ContextValue, Participant, Role, Sink, Turn, clock, trace
from xr_ai_app.context import ContextChannel, render

from ...routing import DestinationResult, RouterDestination
from .controls import CONTROL_QUESTION, CONTROLS, MESSAGES
from .run import Run, plain


class SopRunner(Component):
    """Runs ``procedure`` for each participant who starts it; see the module docstring.

    ``plan`` lists the steps in order, as ``(step id, what the user does)``, so the conversation can tell what
    comes next; without it the summary has only the steps entered so far. ``decide`` classifies control requests
    (Clef or an LLM). ``output`` is where the runner speaks (``None``: the
    app's speech output). ``description`` is what a Router reads to send control requests here. ``controls`` and
    ``messages`` override the default wording (:data:`CONTROLS`, :data:`MESSAGES`) key by key.
    ``escalation_context`` lists the context channels an escalator beside a step may read, by name (``voice.dialogue``
    as ``dialogue``: the person may already have said what it would ask); each escalator picks what it reads (its
    ``context`` setting), the run's summary as ``run``.
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
        escalation_context: Sequence[ContextChannel] = (),
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
        self._escalation_context = list(escalation_context)
        self._budget = context_budget_bytes
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
            focused=self._waiting,
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
            asking=run.asking,
            output=self.output,
            _publish=self._changed,
            _situation=self._situation,
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
        if run.asking:
            lines.append(f"Waiting for the person's answer to: {run.asking}")
        if run.values:
            lines.append("Noted: " + json.dumps({k: plain(v) for k, v in run.values.items()}, ensure_ascii=False))
        return "\n".join(lines)

    def _new_run(self, p: Participant) -> Run:
        previous = self.run(p)
        number = (previous.number + 1) if previous else 1
        run = Run(p, number, name=self.name, output=self.output, _publish=self._changed, _situation=self._situation)
        p.state[self._key] = run
        return run

    async def _situation(self, p: Participant, sources: Sequence[str]) -> str:
        known = self._sources()
        parts = []
        for name in sources:
            if name not in known:
                raise ValueError(f"no context {name!r} for a step; there is {sorted(known)}")
            parts.append(await known[name](p))
        return "\n\n".join(part for part in parts if part)

    def _sources(self) -> dict[str, Callable[[Participant], Awaitable[str]]]:
        """What an escalator may read, by name: ``run`` (this runner's summary) and each ``escalation_context``
        channel (a channel named ``run`` replaces the summary)."""

        async def summary(p: Participant) -> str:
            return self.summary(p)

        def channel(c: ContextChannel) -> Callable[[Participant], Awaitable[str]]:
            async def text(p: Participant) -> str:
                return await render(p, [c], budget_bytes=self._budget)

            return text

        return {"run": summary, **{c.name: channel(c) for c in self._escalation_context}}

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
        """What ``text`` is to the run: a control (``start``, ``stop``, ``skip``, ``restart``), ``answer`` (a reply to
        the question the run is asking, only while one is pending) or ``none``; performs nothing.

        The one decision :attr:`control` makes before it acts, for tests and evaluations. It sees only the
        utterance, whether the guide is running and the pending question, not the run's summary.
        """
        run = self.run(p)
        choices = dict(self._controls)
        # Only what the choice depends on: whether the guide is running, and the question it is waiting on. The run's
        # plan, steps and observations are for the conversation; next to a short utterance they are noise.
        status = "not started" if run is None else run.status
        state = {"request": text, "guide": f"{self.title}: {status}"}
        if run is not None and run.asking:
            state["pending_question"] = run.asking
            choices["answer"] = (
                f"The reply gives the information the assistant's pending question asked for ({run.asking!r}), as a "
                "fact or a guess, or says the person does not know it."
            )
            choices["none"] += (
                " Also anything that does not give the information the pending question asked for: another "
                "question, a remark, or a request."
            )
        return str(await self._decide.decide(self._question, choices=choices, context=json.dumps(state)))

    def _waiting(self, p: Participant) -> bool:
        run = self.run(p)
        return run is not None and run.asking is not None

    async def _on_control(self, turn: Turn, p: Participant) -> DestinationResult:
        op = await self.classify(turn.text, p)
        trace.emit("sop", self.name, op="control", choice=op)
        if op == "answer":
            run = self.run(p)
            if run is not None and run.answer(turn.text):
                return DestinationResult("answer", "")
            op = "none"
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
