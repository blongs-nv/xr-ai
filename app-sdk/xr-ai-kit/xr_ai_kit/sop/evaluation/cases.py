# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a step's cases: the real step, on a recorded scene and a scripted person, scored by its type's scoring.

A case file is about one step::

    step: heat_water
    inputs: {target: {value: 80, unit: celsius}}   # what the step runs with (its inputs_schema), every case
    cases:
      - name: unit_unknown_then_answered
        scene: [26, 346]              # parts of the recording (see .scene)
        person: {answers: [Fahrenheit]}
        expect: {asks: [unit], completes: {after: 330}}   # completes: the step's; asks: its escalator's
      - name: looked_away
        scene: [[26, 36], [5, 12], [5, 12], [5, 12]]
        inputs: {target: {value: 85, unit: celsius}}   # this case's own inputs (over the file's)
        person: {after_tell: [[26, 60]]}
        expect: {says: [kettle, display], completes: false}

The app gives its steps, ``steps(fakes) -> {step id: step}``, the same ones its procedure runs, so the step evaluated
is the step that ships; a case gives the inputs the procedure would pass it (validated against its ``inputs_schema``).
Each case runs on the devkit's virtual-time loop: idle waits are skipped, so a five-minute scene costs what its model
calls cost. How long a model call takes on the scene's clock is chosen with ``--latency``:

- ``real`` (the default): as long as it took on this machine (the result depends on the machine and its load);
- ``zero``: no time (the scene stands still while models answer: is the step right, whatever the models' speed?);
- a profile, ``latency/spark.yaml`` (``{role: seconds}``, :class:`~xr_ai_app.devkit.LatencyProfile`): the time it
  declares for each role, so the same profile gives the same timeline on any machine; ``--latency-scale 2`` runs it
  as a machine twice as slow, to find how much margin a case has.

Every run records how long each role's calls took here (``latency.json``) and writes them as a profile
(``measured-latency.yaml``, the median per role), so a ``real`` run on the target machine becomes its profile. A case
ends when the step finishes or the scene does.

Each step type's scoring is a module next to it (``watch_scoring``, ...) with ``Expect`` (what a case may expect; by
convention an alias, ``Expect = WatchExpect``, so every module has the same name for it) and ``score(trace,
expect)``; the type names it in its ``scoring`` attribute, a tuple of module paths (an app's own step types too), and
:func:`scoring_for` imports it. A step with an escalator beside it names both (the escalator's expects ``asks``,
``says``, ``quiet``), and a case's keys go to the one that owns them. ``scorings={StepType: module}`` overrides that.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, create_model

from ..steps.step_api import Step
from .scene import Recording, SceneFrames
from .scripted_environment import Person, ScriptedEnvironment
from .step_trace import Line, StepTrace


class Scoring(Protocol):
    """How a case is scored (a step type's scoring module is one): ``Expect``, the keys a case may expect, and
    ``score``."""

    Expect: type[BaseModel]

    def score(self, trace: StepTrace, expect: Any) -> list[str]: ...


Scorings = Mapping[type, Scoring | ModuleType]


class Combined:
    """Several scorings of one step (a step and the escalator beside it): a case's ``expect`` keys go to the scoring
    that owns them, and a key none owns is an error."""

    def __init__(self, parts: tuple[Scoring, ...]) -> None:
        self.parts = parts
        fields: dict[str, Any] = {}
        for part in parts:
            for name, info in part.Expect.model_fields.items():
                if name in fields:
                    raise TypeError(f"two scorings expect {name!r}")
                fields[name] = (info.annotation, info)
        self.Expect: type[BaseModel] = create_model(  # the name every scoring has
            "CombinedExpect", __config__=ConfigDict(extra="forbid"), **fields
        )

    def score(self, trace: StepTrace, expect: BaseModel) -> list[str]:
        given = expect.model_dump(exclude_unset=True)
        problems: list[str] = []
        for part in self.parts:
            own = {k: v for k, v in given.items() if k in part.Expect.model_fields}
            problems += part.score(trace, part.Expect.model_validate(own))
        return problems


def scoring_for(step: Step[Any], scorings: Scorings | None = None) -> Scoring:
    """How a case of ``step`` is scored: ``scorings[type(step)]`` when given, else the modules the step names in its
    ``scoring`` attribute, a tuple of module paths (imported here, so running code never loads them; several are
    combined)."""
    kind = type(step)
    if scorings and kind in scorings:
        return scorings[kind]  # type: ignore[return-value]
    names = getattr(step, "scoring", ())
    if not isinstance(names, tuple) or not names or not all(isinstance(n, str) for n in names):
        raise TypeError(
            f"{kind.__name__} names no scoring module; give it a `scoring` attribute, a tuple of module paths"
        )
    parts = tuple(importlib.import_module(n) for n in names)
    return parts[0] if len(parts) == 1 else Combined(parts)  # type: ignore[return-value]


Steps = Callable[[Any], Mapping[str, Step[Any]]]
"""``steps(fakes)``: the app's steps by id (``fakes``: scripted roles for a plumbing run, or ``None``)."""


class Case(BaseModel):
    """One case of a step: a scene, a person, what to expect."""

    model_config = ConfigDict(extra="forbid")
    name: str
    scene: list[Any]
    inputs: dict[str, Any] = {}
    """The step's inputs for this case, over the file's (validated against its ``inputs_schema``)."""
    person: Person = Person()
    conversation: str = ""
    """The conversation so far, one line each (``user: ...``, ``assistant: ...``), as the step and its escalator see it
    (``environment.conversation``); fixed for the case."""
    run_summary: str = ""
    """The run's summary as the step and its escalator see it (``environment.run_summary()``)."""
    expect: dict[str, Any] = {}
    """Validated against the step type's ``Expect``."""
    why: str = ""


class CaseFile(BaseModel):
    """The cases of one step (see the module docstring)."""

    model_config = ConfigDict(extra="forbid")
    step: str
    inputs: dict[str, Any] = {}
    """The step's inputs for every case (a case's own ``inputs`` add to them)."""
    cases: list[Case]

    @classmethod
    def load(cls, path: Path | str) -> CaseFile:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


@dataclass(slots=True)
class CaseResult:
    step: str
    case: Case
    trace: StepTrace
    problems: list[str] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.problems and self.trace.error is None

    def as_json(self) -> dict[str, Any]:
        t = self.trace
        return {
            "step": self.step,
            "case": self.case.name,
            "ok": self.ok,
            "problems": self.problems,
            "why": self.case.why,
            "completed": t.completed,
            "completed_at_s": _round2(t.completed_at_s),
            "error": t.error,
            "lines": [
                {"t_s": _round2(x.t_s), "by": x.by, "kind": x.kind, "text": x.text, "answer": x.answer} for x in t.lines
            ],
            "states": [{"t_s": _round2(at), "state": _summary(state)} for at, state in t.states],
            "wall_s": _round2(self.wall_s),
        }


def inputs_for(step: Step[Any], values: Mapping[str, Any]) -> Any:
    """The inputs a case gives ``step``: ``values`` validated against its ``inputs_schema`` (``None`` without one)."""
    if step.inputs_schema is None:
        if values:
            raise ValueError(f"step {step.id!r} takes no inputs; the case gives {sorted(values)}")
        return None
    return step.inputs_schema.model_validate(values)


async def run_case(
    step: Step[Any],
    case: Case,
    recording: Recording,
    *,
    inputs: Mapping[str, Any] | None = None,
    scorings: Scorings | None = None,
) -> CaseResult:
    """Run ``step`` with ``inputs`` (default: the case's) on the case's scene and person until it finishes or the scene
    ends; score it."""
    scoring = scoring_for(step, scorings)
    step_inputs = inputs_for(step, case.inputs if inputs is None else inputs)
    expect = scoring.Expect.model_validate(case.expect)
    scene = SceneFrames(recording, case.scene)
    environment = ScriptedEnvironment(
        step.id, scene, case.person, conversation=case.conversation, run_summary=case.run_summary
    )
    began = time.monotonic()
    scene.begin()
    task = asyncio.ensure_future(step.run(environment, step_inputs))
    while not task.done() and (remaining := scene.remaining_s()) > 0:
        await asyncio.wait({task}, timeout=remaining)
    trace = environment.trace
    if task.done():
        if task.cancelled():
            trace.error = "cancelled"
        elif (exc := task.exception()) is not None:
            trace.error = repr(exc)
        else:
            trace.completed, trace.completed_at_s, trace.result = True, scene.now(), task.result()
            if done_line := step.done_line(step_inputs, trace.result):  # what the runner would say as it finishes
                trace.lines.append(Line(scene.now(), "done", done_line))
    else:
        task.cancel()
        await asyncio.wait({task})  # let the step end; the scene is over, so its outcome no longer counts
    problems = [] if trace.error else scoring.score(trace, expect)
    return CaseResult(step.id, case, trace, problems, time.monotonic() - began)


async def run_files(
    files: Sequence[Path],
    steps: Steps,
    recording: Recording,
    *,
    fakes: Any = None,
    only: Sequence[str] = (),
    concurrency: int = 4,
    scorings: Scorings | None = None,
) -> list[CaseResult]:
    """Run every case of ``files`` (``only``: step ids or case names), ``concurrency`` at a time."""
    gate = asyncio.Semaphore(concurrency)
    by_id = steps(fakes)
    jobs = []
    for path in files:
        spec = CaseFile.load(path)
        if spec.step not in by_id:
            raise KeyError(f"{path.name}: no step {spec.step!r}; the app's steps are {sorted(by_id)}")
        for case in spec.cases:
            if only and spec.step not in only and case.name not in only:
                continue
            step, inputs = by_id[spec.step], {**spec.inputs, **case.inputs}

            async def one(step: Step[Any] = step, case: Case = case, inputs: dict[str, Any] = inputs) -> CaseResult:
                async with gate:
                    return await run_case(step, case, recording, inputs=inputs, scorings=scorings)

            jobs.append(one())
    return list(await asyncio.gather(*jobs))


def latency_stats(real_latencies_s: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    """Per role: how many calls, and the median and 95th-percentile time they took, in seconds."""
    stats = {}
    for role, times in sorted(real_latencies_s.items()):
        ordered = sorted(times)
        stats[role] = {
            "calls": len(ordered),
            "p50_s": round(ordered[len(ordered) // 2], 3),
            "p95_s": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        }
    return stats


def report(
    results: Sequence[CaseResult], *, latency: str = "real", measured: Mapping[str, Mapping[str, float]] | None = None
) -> str:
    """The results as Markdown: totals, the model latency used and measured, then one table per step."""
    passed = sum(r.ok for r in results)
    lines = ["# Step cases", "", f"**{passed} of {len(results)} cases passed.**", ""]
    lines.append(f"Model latency on the scene's clock: {latency}.")
    if measured:
        took = ", ".join(
            f"{role} {m['p50_s']:g} s (p95 {m['p95_s']:g} s, {m['calls']:g} calls)" for role, m in measured.items()
        )
        lines.append(f"Measured here (median): {took}.")
    lines.append("")
    by_step: dict[str, list[CaseResult]] = {}
    for r in results:
        by_step.setdefault(r.step, []).append(r)
    for step, rows in by_step.items():
        lines += [f"## {step}", "", "| case | ok | completed | said | problems |", "|---|---|---|---|---|"]
        for r in rows:
            t = r.trace
            done = f"at {t.completed_at_s:.0f} s" if t.completed and t.completed_at_s is not None else "no"
            said = "; ".join(f"{x.by} {x.kind} {x.t_s:.0f} s: {x.text}" for x in t.lines) or ""
            wrong = "; ".join(r.problems) or (t.error or "")
            lines.append(f"| {r.case.name} | {'yes' if r.ok else '**no**'} | {done} | {said} | {wrong} |")
        lines.append("")
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None,
    *,
    steps: Steps,
    cases: Path,
    fakes_from: Callable[[Path], Any],
    scorings: Scorings | None = None,
) -> int:
    """The command line an app's evaluation script calls; the exit code is 0 when every case passed.

    ``cases`` is the folder of case files (one per step); ``fakes_from`` loads scripted roles for ``--fakes``;
    ``scorings`` scores the app's own step types.
    """
    parser = argparse.ArgumentParser(description="Run step cases on a recording.")
    parser.add_argument("--frames-index", type=Path, required=True, help="the recording's frame index CSV")
    parser.add_argument("--only", nargs="*", default=[], help="step ids or case names")
    parser.add_argument("--out", type=Path, default=Path("runs/steps"))
    parser.add_argument("--fakes", type=Path, help="scripted roles (a plumbing run; the numbers mean nothing)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--latency", default="real", help="model latency on the scene's clock: real, zero, or a profile YAML"
    )
    parser.add_argument(
        "--latency-scale", type=float, default=1.0, help="multiply a profile's times (2: twice as slow)"
    )
    args = parser.parse_args(argv)
    from xr_ai_app.devkit import LatencyProfile, run_virtual  # noqa: PLC0415 - only when evaluating

    files = sorted(cases.glob("*.yaml"))
    fakes = fakes_from(args.fakes) if args.fakes else None
    recording = Recording.load(args.frames_index)
    if args.latency == "real":
        profile, latency = None, "real (as long as each call took on this machine)"
    elif args.latency == "zero":
        profile, latency = LatencyProfile.zero(), "zero (the scene stands still while models answer)"
    else:
        profile = LatencyProfile.load(args.latency, scale=args.latency_scale)
        latency = f"the profile {args.latency}" + (
            f", scaled {args.latency_scale:g}x" if args.latency_scale != 1 else ""
        )

    async def run_and_measure() -> tuple[list[CaseResult], dict[str, list[float]]]:
        results = await run_files(
            files, steps, recording, fakes=fakes, only=args.only, concurrency=args.concurrency, scorings=scorings
        )
        return results, dict(getattr(asyncio.get_running_loop(), "real_latencies_s", {}))

    results, real_latencies_s = run_virtual(run_and_measure(), latency=profile)
    measured = latency_stats(real_latencies_s)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps([r.as_json() for r in results], indent=2), encoding="utf-8")
    (args.out / "latency.json").write_text(json.dumps({"latency": latency, "measured": measured}, indent=2), "utf-8")
    if measured:
        medians = "".join(f"{role}: {m['p50_s']:g}\n" for role, m in measured.items())
        (args.out / "measured-latency.yaml").write_text(
            f"# The median time each role's calls took on this machine; use with --latency.\n{medians}", "utf-8"
        )
    text = report(results, latency=latency, measured=measured)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0 if all(r.ok for r in results) else 1


def _summary(state: Any) -> str:
    """A reported state as text, as ``Run.report_progress`` records it (its ``summary()``, if it has one)."""
    summary = getattr(state, "summary", None)
    return summary() if callable(summary) else str(state)


def _round2(value: float | None) -> float | None:
    return None if value is None else round(value, 2)
