# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One run of a procedure for one participant: what the procedure function receives and what a step touches.

A :class:`Run` is what a step touches (a :class:`~..steps.step_api.StepEnvironment`), and so what an escalator
beside a step touches too; :class:`~.sop_runner.SopRunner` creates one per start.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from xr_ai_app import Participant, Sink

from ..steps.conversation import Conversation, FixedConversation
from ..steps.step_api import Frames, Step, speaker

T = TypeVar("T")

StepStatus = Literal["running", "done", "skipped", "failed"]
RunStatus = Literal["running", "completed", "stopped", "failed"]


@dataclass(slots=True)
class StepRecord:
    """One step the run has entered."""

    id: str
    status: StepStatus
    detail: str = ""
    """The step's latest reported progress (:meth:`Run.report_progress`), as text."""
    skippable: bool = True
    """Whether "skip" may cancel it (a step that does not wait on the person is not)."""


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

    asking: str | None = None
    """The question the run is waiting for the person to answer (see :meth:`ask`), if any."""

    output: Sink | None = field(default=None, repr=False)
    _publish: Callable[..., Awaitable[None]] | None = field(default=None, repr=False)
    conversation: Conversation = field(default_factory=FixedConversation, repr=False)
    """The conversation with the person (the runner's ``conversation``; none without one)."""
    _summary: Callable[[], str] | None = field(default=None, repr=False)
    _answer: asyncio.Future[str] | None = field(default=None, repr=False)

    @property
    def current(self) -> StepRecord | None:
        """The step running now, if any."""
        if self.steps and self.steps[-1].status == "running":
            return self.steps[-1]
        return None

    @property
    def frames(self) -> Frames:
        """The participant's camera frames."""
        return self.p.frames

    async def step(
        self,
        step: Step[T] | str,
        inputs: Any = None,
        *,
        work: Callable[[], Awaitable[T]] | None = None,
        start_line: str | None = None,
    ) -> T | None:
        """Run one step as its own task with ``inputs``; return its result, or ``None`` when the user skipped it.

        ``step`` is a step object (``run.step(steep, SteepInputs(duration_s=120))``: its ``run(self, inputs)``
        is the work; ``inputs`` is an instance of its ``inputs_schema``, or ``None`` when it takes none) or a step id
        with ``work``, any coroutine function of the app's own. The step's ``start_line(inputs)`` is spoken as it
        starts (``start_line`` here replaces it) and its ``done_line(inputs, result)`` when it finishes (not when it
        is skipped). A step that fails fails the run (the exception propagates).
        """
        step_object: Step[T] | None = None
        skippable = True
        if isinstance(step, str):
            if work is None:
                raise TypeError("run.step(step_id, work=...): work is required with a step id")
            step_id = step
        else:
            step_object = step
            step_id = step.id
            skippable = getattr(step, "skippable", True)
            start_line = start_line if start_line is not None else step.start_line(inputs)
            work = lambda: step.run(self, inputs)  # noqa: E731
        record = StepRecord(step_id, "running", skippable=skippable)
        self.steps.append(record)
        await self._changed("step", step=step_id, status="running")
        if start_line:
            await self.say(start_line)
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
        if step_object is not None and (done_line := step_object.done_line(inputs, result)):
            await self.say(done_line)
        return result

    async def report_progress(self, step_id: str, state: Any) -> None:
        """Record a step's latest state as its detail (the state's ``summary()``, if it has one)."""
        record = next((s for s in reversed(self.steps) if s.id == step_id), None)
        if record is None:
            return
        summary = getattr(state, "summary", None)
        record.detail = summary() if callable(summary) else str(state)
        await self._changed("state", step=record.id, detail=record.detail)

    async def note(self, **values: Any) -> None:
        """Record facts for the conversation and the record (for example the brewing settings in use)."""
        self.values.update(values)
        await self._changed("note", **{k: plain(v) for k, v in values.items()})

    async def say(self, text: str) -> None:
        """Speak on the assistant's own initiative. Lines queue in order: a step's ``done_line`` and the next step's
        ``start_line`` are both spoken, however close together."""
        await self.p.say(text, to=self.output)

    async def ask(self, question: str, *, timeout_s: float = 60.0) -> str | None:
        """Ask the person ``question`` and wait for the answer; ``None`` when none came within ``timeout_s``.

        While the question is pending the runner holds the router's focus, so the next turns reach it first: an
        answer (including "I don't know") returns here, a control (stop, skip) still works (skipping cancels the
        step and with it the question), and anything else passes on to the conversation while the question stays
        pending. The question and the answer are in the run's summary and record.
        """
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.asking, self._answer = question, future
        await self._changed("ask", question=question, by=speaker())
        await self.say(question)
        answer: str | None = None
        try:
            answer = await asyncio.wait_for(asyncio.shield(future), timeout_s)
        except TimeoutError:
            answer = None
        finally:
            self.asking, self._answer = None, None
            if not future.done():
                future.cancel()
        await self._changed("answer", question=question, answer=answer)
        return answer

    async def run_summary(self) -> str:
        """The run as text, as the conversation sees it: its steps, their states, what was noted."""
        return self._summary() if self._summary is not None else ""

    def answer(self, text: str) -> bool:
        """Give the pending question its answer; whether one was pending."""
        if self._answer is None or self._answer.done():
            return False
        self._answer.set_result(text)
        return True

    async def _changed(self, op: str, **data: Any) -> None:
        if self._publish is not None:
            await self._publish(self, op, **data)


def plain(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    return value if isinstance(value, (str, int, float, bool, type(None), list, dict)) else str(value)
