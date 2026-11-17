# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One run of a procedure for one participant: what the procedure function receives and what a step touches.

A :class:`Run` is a :class:`~..step.StepIO`; :class:`~.runner.SopRunner` creates one per start.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from xr_ai_app import Participant, Sink

from ..step import Frames, Step

T = TypeVar("T")

StepStatus = Literal["running", "done", "skipped", "failed"]
RunStatus = Literal["running", "completed", "stopped", "failed"]


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

    asking: str | None = None
    """The question the run is waiting for the person to answer (see :meth:`ask`), if any."""

    output: Sink | None = field(default=None, repr=False)
    _publish: Callable[..., Awaitable[None]] | None = field(default=None, repr=False)
    _situation: Callable[[Participant, Sequence[str]], Awaitable[str]] | None = field(default=None, repr=False)
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
        work: Callable[[], Awaitable[T]] | None = None,
        *,
        say: str | None = None,
    ) -> T | None:
        """Run one step as its own task; return its result, or ``None`` when the user skipped it.

        ``step`` is a step object (``run.step(heat)``: its ``run(self)`` is the work and its ``say`` the instruction)
        or a step id with ``work``, any coroutine function of the app's own. ``say`` is spoken as the step starts
        (it replaces the step's own). A step that fails fails the run (the exception propagates).
        """
        if isinstance(step, str):
            if work is None:
                raise TypeError("run.step(step_id, work): work is required with a step id")
            step_id = step
        else:
            step_id, say, obj = step.id, say if say is not None else step.say, step
            work = lambda: obj.run(self)  # noqa: E731
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
        await self._changed("note", **{k: plain(v) for k, v in values.items()})

    async def say(self, text: str) -> None:
        """Speak on the assistant's own initiative, tagged with the procedure's name (a newer line replaces it)."""
        await self.p.say(text, source=f"sop:{self.name}", to=self.output)

    async def ask(self, question: str, *, timeout_s: float = 60.0) -> str | None:
        """Ask the person ``question`` and wait for the answer; ``None`` when none came within ``timeout_s``.

        While the question is pending the runner holds the router's focus, so the next turns reach it first: an
        answer (including "I don't know") returns here, a control (stop, skip) still works (skipping cancels the
        step and with it the question), and anything else passes on to the conversation while the question stays
        pending. The question and the answer are in the run's summary and record.
        """
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.asking, self._answer = question, future
        await self._changed("ask", question=question)
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

    async def situation(self, sources: Sequence[str]) -> str:
        """What is going on, for a step's escalator: ``run`` (the run's summary) and the runner's ``step_context``
        channels by name, in the order given; a name the runner does not have is an error."""
        return await self._situation(self.p, sources) if self._situation is not None else ""

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
