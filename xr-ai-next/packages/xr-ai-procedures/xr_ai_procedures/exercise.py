# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercising procedures on a clip, through their real code: the devkit targets ``step:`` and ``procedure:``.

``step:<id>`` runs ONE step. A step needs nothing a test has to fake: :class:`~.step.StepDef` builds it from the
participant (its camera), the :class:`~.run.Run` (the facts the user told each step, and the results of the steps
before it) and its SOP entry, and ``step.run(report)`` ends in ``Done``, ``Needs`` or ``Stuck``. The harness builds
it the way the procedure does, with facts and earlier results injected::

    xr-ai-dev exercise app.py step:bed_clear --video demo.mp4 --from 12 --to 31.5 --expect Done --between 14 30
    xr-ai-dev exercise app.py step:heat_water --video demo.mp4 --from 40 --to 95 \\
        --option 'facts={"identify": {"tea": "green"}}' --option 'results={"identify": "green tea"}'

``procedure:`` (or ``procedure:<name>``) runs the WHOLE procedure on a recording: started in code (no user turn, no
decision model), every step's outcome and the clip time it happened recorded, and, with annotations
(``{step_id: [earliest, latest]}``), each step checked against its window: ``ok``, ``early``, ``late`` or
``missing``. It stops when the run ends, when the procedure asks the user something (nobody answers in a test), or
soon after the clip ends::

    xr-ai-dev exercise app.py procedure: --video demo.mp4 --option annotations=@demo.windows.yaml

In Python: ``await exercise(app, step_unit(procedure, "bed_clear"), video=...)`` or ``procedure_unit(procedure,
annotations=...)``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .guide import GuidedProcedure
from .run import Run

if TYPE_CHECKING:
    from .events import ProcedureEvent

if TYPE_CHECKING:
    from xr_ai_next import App, Participant
    from xr_ai_next.devkit import Probe, Unit

__all__ = [
    "ProcedureRun",
    "StepTiming",
    "procedure_unit",
    "procedures_of",
    "resolve_procedure",
    "resolve_step",
    "step_unit",
]


def step_unit(
    procedure: GuidedProcedure,
    step_id: str,
    *,
    facts: Mapping[str, Mapping[str, Any]] | None = None,
    results: Mapping[str, Any] | None = None,
) -> Unit:
    """A devkit unit that builds step ``step_id`` as ``procedure`` would and runs it.

    ``facts``: what the user told each step, ``{step_id: {fact: value}}`` (validated by that step's facts model).
    ``results``: earlier steps' results, ``{step_id: value}``. The step's report lines go to ``probe.report``.
    """
    if step_id not in procedure.defs:
        raise ValueError(f"{procedure.name!r} has no step {step_id!r}; its steps: {', '.join(procedure.defs)}")
    unknown = set(facts or {}) - set(procedure.fact_models)
    if unknown:
        with_facts = sorted(procedure.fact_models)
        raise ValueError(f"no facts model for steps {sorted(unknown)}; steps with facts: {with_facts}")
    definition, spec = procedure.defs[step_id], procedure.sop.step(step_id)

    async def unit(p: Participant, probe: Probe) -> Any:
        told = facts or {}
        run = Run(facts={sid: model(**told.get(sid, {})) for sid, model in procedure.fact_models.items()})
        run.results.update(results or {})
        run.index = [s.id for s in procedure.sop.steps].index(step_id)
        step, _key = definition.build(p, run, spec)
        return await step.run(probe.report)

    return unit


def procedures_of(app: App) -> list[GuidedProcedure]:
    """The app's guided procedures, in the order they were added."""
    from xr_ai_next.devkit import components  # noqa: PLC0415 - the devkit is only needed when exercising

    return components(app, GuidedProcedure)


def resolve_step(app: App, name: str, options: Mapping[str, Any]) -> Unit:
    """``step:<step id>`` (or ``step:<procedure name>/<step id>`` when the app has several procedures).

    Options: ``facts`` and ``results``, as in :func:`step_unit`.
    """
    found = procedures_of(app)
    procedure_name, _, step_id = name.rpartition("/")
    if procedure_name:
        found = [proc for proc in found if proc.name == procedure_name]
    if len(found) != 1:
        names = [proc.name for proc in procedures_of(app)]
        raise ValueError(f"step:{name}: expected one procedure, the app has {names or 'none'} (use <procedure>/<step>)")
    return step_unit(found[0], step_id, facts=options.get("facts"), results=options.get("results"))


# ------------------------------------------------------------------------------------ the whole procedure

Verdict = Literal["ok", "early", "late", "missing"]


@dataclass
class StepTiming:
    """One step of an exercised run: how it ended, when (clip seconds), and against its annotated window."""

    step: str
    outcome: str = "not reached"
    """``done``, ``confirmed``, ``skipped``, ``stuck``, ``needs``, ``failed``, ``running`` or ``not reached``."""
    started_at_s: float | None = None
    ended_at_s: float | None = None
    window: tuple[float, float] | None = None
    verdict: Verdict | None = None
    """Against ``window``: done inside it (``ok``), before (``early``), after (``late``), or not done (``missing``)."""


@dataclass
class ProcedureRun:
    """What a procedure did on a clip (the result of a ``procedure:`` exercise)."""

    status: str
    steps: list[StepTiming]
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Every annotated step is ``ok``; without annotations, the run completed."""
        judged = [s for s in self.steps if s.window is not None]
        return all(s.verdict == "ok" for s in judged) if judged else self.status == "completed"


def procedure_unit(
    procedure: GuidedProcedure,
    *,
    annotations: Mapping[str, Any] | None = None,
    facts: Mapping[str, Mapping[str, Any]] | None = None,
) -> Unit:
    """A devkit unit that runs ``procedure`` from its first step and returns a :class:`ProcedureRun`.

    ``annotations``: ``{step_id: [earliest, latest]}`` in clip seconds. ``facts``: what the user would have told each
    step, ``{step_id: {fact: value}}``, set when the run starts.
    """
    ids = [s.id for s in procedure.sop.steps]
    windows = {str(k): (float(v[0]), float(v[1])) for k, v in (annotations or {}).items()}
    unknown = set(windows) - set(ids)
    if unknown:
        raise ValueError(f"annotations for unknown steps {sorted(unknown)}; the steps: {ids}")

    async def unit(p: Participant, probe: Probe) -> ProcedureRun:
        timings = {sid: StepTiming(sid, window=windows.get(sid)) for sid in ids}
        events: list[dict[str, Any]] = []
        asked = asyncio.Event()

        def record(event: ProcedureEvent, who: Participant) -> None:
            if who.id != p.id:
                return
            at = probe.video_time()
            events.append({"kind": event.kind, "step": event.step, "at_s": at, **_plain(event.data)})
            timing = timings.get(event.step or "")
            if event.kind == "run_start" and facts:
                run = procedure.run_of(p)
                for sid, told in facts.items():
                    run.facts[sid] = run.facts[sid].model_copy(update=dict(told))
            elif timing is None:
                return
            elif event.kind == "step_start":
                timing.outcome, timing.started_at_s = "running", at
            elif event.kind == "step_done":
                timing.outcome, timing.ended_at_s = str(event.data.get("how")), at
            elif event.kind in ("step_stuck", "step_needs", "step_failed"):
                timing.outcome = event.kind.removeprefix("step_")
            elif event.kind == "back":
                for later in ids[ids.index(event.step) :]:
                    timings[later] = StepTiming(later, window=windows.get(later))
            if event.kind == "ask":
                asked.set()

        procedure.on("*", record)
        try:
            run = await procedure.start(p)
            waits = [asyncio.ensure_future(e.wait()) for e in (run.finished, asked)]
            clip = asyncio.ensure_future(_after(probe.clip_ended, probe.linger_s / 2))
            await asyncio.wait([*waits, clip], return_when=asyncio.FIRST_COMPLETED)
            for task in (*waits, clip):
                task.cancel()
            if run.status == "running":
                await procedure.stop(p)
            for timing in timings.values():
                timing.verdict = _verdict(timing)
            return ProcedureRun(run.status, list(timings.values()), events)
        finally:
            procedure.off(record)

    return unit


async def _after(event: asyncio.Event, delay_s: float) -> None:
    await event.wait()
    await asyncio.sleep(delay_s)


def _verdict(timing: StepTiming) -> Verdict | None:
    if timing.window is None:
        return None
    earliest, latest = timing.window
    if timing.outcome not in ("done", "confirmed") or timing.ended_at_s is None:
        return "missing"
    if timing.ended_at_s < earliest:
        return "early"
    return "late" if timing.ended_at_s > latest else "ok"


def _plain(data: Mapping[str, Any]) -> dict[str, Any]:
    """Event data as JSON-friendly values (a model as its fields, anything else unknown as its repr)."""
    out = {}
    for key, value in data.items():
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            out[key] = dump()
        elif value is None or isinstance(value, (str, int, float, bool, list, dict)):
            out[key] = value
        else:
            out[key] = repr(value)
    return out


def resolve_procedure(app: App, name: str, options: Mapping[str, Any]) -> Unit:
    """``procedure:`` (the app's only procedure) or ``procedure:<name>``. Options: ``annotations``, ``facts``."""
    found = [proc for proc in procedures_of(app) if not name or proc.name == name]
    if len(found) != 1:
        names = [proc.name for proc in procedures_of(app)]
        raise ValueError(f"procedure:{name}: expected one procedure, the app has {names or 'none'}")
    return procedure_unit(found[0], annotations=options.get("annotations"), facts=options.get("facts"))
