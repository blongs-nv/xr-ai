# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The supervisor of a :class:`~.guide.GuidedProcedure` run: one background task per run, its only writer.

It plays the steps in order. For each it speaks the ``enter`` line, builds the step from the run, runs it in its own
task (``procedure:<name>:step``) and applies, in order, what was posted to the run's inbox: what the step sees and
how it ended (tagged with the build's generation, so a replaced step's late posts are ignored), and the controls
(stop, next, back, facts). When a step cannot go on, the SOP's policy decides; the step never does. Trace events:
``procedure`` with names ``run_start``, ``step_start``, ``step_done``, ``step_stuck``, ``step_needs``,
``step_failed``, ``ask``, ``back`` and ``run_end``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xr_ai_next import trace

from .events import ProcedureEvent
from .perception import Done, Needs, Stuck
from .run import Failed, Run, Stopped, cancel

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Hashable

    from xr_ai_next import Participant

    from .guide import GuidedProcedure
    from .perception import Runnable
    from .spec import StepSpec


class Supervisor:
    """Drives runs of one procedure (see the module docstring)."""

    def __init__(self, procedure: GuidedProcedure) -> None:
        self._proc = procedure

    async def _emit(self, p: Participant, name: str, step: str | None = None, **data: Any) -> None:
        """Trace the transition and tell the procedure's listeners."""
        trace.emit("procedure", name, procedure=self._proc.name, **({"step": step} if step else {}), **data)
        event = ProcedureEvent(name, self._proc.name, p.id, step, data)  # type: ignore[arg-type]
        await self._proc._dispatch(event, p)

    async def drive(self, p: Participant, run: Run) -> None:
        """Play the run to its end (the background task ``procedure:<name>``)."""
        proc = self._proc
        await self._emit(p, "run_start", steps=[s.id for s in proc.sop.steps])
        try:
            while run.index < len(proc.sop.steps):
                await self._play(p, run, proc.sop.steps[run.index])
            run.status = "completed"
            await self._speak(p, run, proc.message(run, "complete"))
        except Stopped:
            run.status = "stopped"
        except Exception as exc:  # noqa: BLE001 - the run is the unit of failure; the user hears it stopped
            run.status = "failed"
            trace.emit("error", proc.name, where="procedure", error=f"{type(exc).__name__}: {exc}")
            await self._speak(p, run, proc.message(run, "failed"))
        finally:
            if run.status == "running":
                run.status = "stopped"  # cancelled: the participant left, or a new run replaced this one
            run.pending = None
            run.finished.set()
            if proc.run_of(p) is run:  # a newer run owns the summary otherwise
                await proc.publish(p, run)
            await self._emit(p, "run_end", status=run.status, outcomes=dict(run.outcomes))

    async def _play(self, p: Participant, run: Run, spec: StepSpec) -> None:
        """One step, until it is over (done, skipped, or the user went back)."""
        proc = self._proc
        run.seen, run.pending = "", None
        await self._emit(p, "step_start", spec.id, index=run.index)
        await proc.publish(p, run)
        await self._speak(p, run, proc.line(run, spec.enter))
        again = True
        while again:
            step, key = proc.defs[spec.id].build(p, run, spec)
            run.generation += 1
            task = p.spawn(self._execute(run, step, run.generation), key=f"procedure:{proc.name}:step", supersede=True)
            try:
                again = await self._watch(p, run, spec, task, key)
            finally:
                task.cancel()

    async def _execute(self, run: Run, step: Runnable, generation: int) -> None:
        """Run a step in its own task; post what it sees and how it ended."""

        async def report(text: str) -> None:
            run.post("seen", (generation, text))

        try:
            outcome: Any = await step.run(report)
        except Exception as exc:  # noqa: BLE001 - a step that raised ends in Failed; the SOP's policy applies
            outcome = Failed(f"{type(exc).__name__}: {exc}")
        run.post("outcome", (generation, outcome))

    async def _watch(self, p: Participant, run: Run, spec: StepSpec, task: asyncio.Task[Any], key: Hashable) -> bool:
        """Apply the inbox until the step is over (``False``) or must be built again (``True``)."""
        proc = self._proc
        generation = run.generation
        while True:
            if not run.inbox:
                run.wake.clear()
                await run.wake.wait()
                continue
            command, payload = run.inbox.popleft()
            if command in ("seen", "outcome"):
                posted, value = payload
                if posted != generation:
                    continue  # from a step that was replaced
                if command == "seen":
                    run.seen = value
                    await proc.publish(p, run)
                    continue
                if isinstance(value, Done):
                    await self._finish(p, run, spec, "done", value.value)
                    return False
                policy = await self._unable(p, run, spec, value)
                if policy is not None:
                    return policy
            elif command == "stop":
                await cancel(task)
                raise Stopped
            elif command == "next":
                await cancel(task)
                await self._finish(p, run, spec, "skipped")
                return False
            elif command == "back":
                await cancel(task)
                await self._go_back(p, run)
                return False
            elif command == "facts":
                self._apply(run, payload)
                await proc.publish(p, run)
                if payload.step_done:
                    await cancel(task)  # the user says it is done
                    await self._finish(p, run, spec, "confirmed")
                    return False
                # Any fact may change this step (it can read other steps' facts), so its key is computed again.
                _, new_key = proc.defs[spec.id].build(p, run, spec)
                if run.pending is not None or new_key != key:
                    await cancel(task)
                    run.pending = None
                    return True

    async def _unable(self, p: Participant, run: Run, spec: StepSpec, outcome: Any) -> bool | None:
        """A step could not go on. ``True``: build it again; ``False``: it is over; ``None``: wait for the user."""
        proc = self._proc
        if isinstance(outcome, Needs):  # only the user can give it
            await self._emit(p, "step_needs", spec.id, fact=outcome.fact)
            await self._ask(p, run, outcome.fact, proc.line(run, proc.question(spec, outcome.fact)))
            return None
        if isinstance(outcome, Stuck):
            await self._emit(p, "step_stuck", spec.id, reason=outcome.reason, policy=spec.unable)
        else:
            error = outcome.error if isinstance(outcome, Failed) else repr(outcome)
            await self._emit(p, "step_failed", spec.id, error=error, policy=spec.unable)
        if spec.unable == "skip":
            await self._finish(p, run, spec, "skipped")
            run.carry = " ".join(x for x in (proc.message(run, "gave_up"), run.carry) if x)
            return False
        if spec.unable == "confirm":
            question = proc.line(run, spec.confirm)
            await self._ask(p, run, "confirm", proc.line(run, proc.question(spec, "confirm"), question=question))
            return None
        if isinstance(outcome, Stuck):  # show: ask to show it, and keep looking
            await self._speak(p, run, proc.line(run, proc.question(spec, "view"), instruction=spec.instruction))
            return True
        await self._ask(p, run, "retry", proc.line(run, proc.question(spec, "retry")))
        return None

    async def _ask(self, p: Participant, run: Run, kind: str, text: str) -> None:
        """Ask and wait: the next turn goes straight to the controls (focus) and is read as the answer."""
        await self._speak(p, run, text)
        run.pending, run.asked = kind, text
        await self._emit(p, "ask", self._proc.sop.steps[run.index].id, about=kind, text=text)
        await self._proc.publish(p, run)

    def _apply(self, run: Run, told: Any) -> None:
        """Hand each fact to the step that owns it; facts for a step ahead wait there until it is built."""
        by_step: dict[str, dict[str, Any]] = {}
        for name, value in told.model_dump(exclude_none=True, exclude={"step_done"}).items():
            by_step.setdefault(self._proc.owner[name], {})[name] = value
        for step_id, update in by_step.items():
            run.facts[step_id] = run.facts[step_id].model_copy(update=update)

    async def _finish(self, p: Participant, run: Run, spec: StepSpec, how: str, value: Any = None) -> None:
        run.outcomes[spec.id] = how
        if how == "done" and value is not None:
            run.results[spec.id] = value
        line = spec.skipped if how == "skipped" else spec.done
        run.carry = self._proc.line(run, line) if line else ""
        run.pending = None
        run.index += 1
        await self._emit(p, "step_done", spec.id, how=how, value=value)

    async def _go_back(self, p: Participant, run: Run) -> None:
        """Back one step; it and every later step are undone, so a repeated step feeds new results forward."""
        run.index = max(0, run.index - 1)
        for spec in self._proc.sop.steps[run.index :]:
            run.results.pop(spec.id, None)
            run.outcomes.pop(spec.id, None)
        run.carry, run.pending = "", None
        await self._emit(p, "back", self._proc.sop.steps[run.index].id)

    async def _speak(self, p: Participant, run: Run, text: str) -> None:
        """Say the carried words and ``text`` as one utterance (queued updates from one source replace each other)."""
        full = " ".join(part for part in (run.carry, text) if part)
        run.carry = ""
        if full:
            await p.say(full, to=self._proc.output, source=f"procedure:{self._proc.name}")
