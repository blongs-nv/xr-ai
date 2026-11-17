# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a step's cases: the real step, on a recorded scene and a scripted person, scored by the step's own ``score``.

A case file is about one step::

    step: heat_water
    build: {temperature_c: 80}        # passed to the app's step factory (run-time targets, settings)
    cases:
      - name: unit_unknown_then_answered
        scene: [26, 346]              # parts of the recording (see .scene)
        person: {answers: [Fahrenheit]}
        expect: {asks: [unit], completes: {after: 330}}   # the step type's Expect
      - name: looked_away
        scene: [[26, 36], [5, 12], [5, 12], [5, 12]]
        build: {unit: fahrenheit}     # this case's own factory values
        person: {after_tell: [[26, 60]]}
        expect: {says: [kettle, display], completes: false}

The app gives a factory, ``build(values, fakes) -> {step id: step}``, the same one its procedure builds its steps
with, so the step evaluated is the step that ships. Each case runs on the devkit's virtual-time loop: idle waits are
skipped and model calls take their real time, so a five-minute scene costs what its model calls cost, and their
latency counts as it would live. A case ends when the step finishes or the scene does.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from ..step import Step, StepTrace
from .io import Person, ScriptedIO
from .scene import Recording, SceneFrames

Build = Callable[[Mapping[str, Any], Any], Mapping[str, Step[Any]]]
"""``build(values, fakes)``: the app's steps for these factory values (``fakes``: scripted roles, or ``None``)."""


class Case(BaseModel):
    """One case of a step: a scene, a person, what to expect."""

    model_config = ConfigDict(extra="forbid")
    name: str
    scene: list[Any]
    build: dict[str, Any] = {}
    person: Person = Person()
    situation: str = ""
    """What the app would say is going on around the step (live: the run's summary and the dialogue)."""
    expect: dict[str, Any] = {}
    """Validated against the step type's ``Expect``."""
    why: str = ""


class CaseFile(BaseModel):
    """The cases of one step (see the module docstring)."""

    model_config = ConfigDict(extra="forbid")
    step: str
    build: dict[str, Any] = {}
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
            "completed_at_s": _r(t.completed_at_s),
            "error": t.error,
            "lines": [{"t_s": _r(x.t_s), "kind": x.kind, "text": x.text, "answer": x.answer} for x in t.lines],
            "states": [{"t_s": _r(at), "state": _summary(state)} for at, state in t.states],
            "wall_s": _r(self.wall_s),
        }


async def run_case(step: Step[Any], case: Case, recording: Recording) -> CaseResult:
    """Run ``step`` on the case's scene and person until it finishes or the scene ends; score it."""
    kind = type(step)
    expect = kind.Expect.model_validate(case.expect)  # type: ignore[attr-defined]
    scene = SceneFrames(recording, case.scene)
    io = ScriptedIO(step.id, scene, case.person, case.situation)
    began = time.monotonic()
    scene.begin()
    task = asyncio.ensure_future(step.run(io))
    while not task.done() and (remaining := scene.remaining_s()) > 0:
        await asyncio.wait({task}, timeout=remaining)
    trace = io.trace
    if task.done():
        if task.cancelled():
            trace.error = "cancelled"
        elif (exc := task.exception()) is not None:
            trace.error = repr(exc)
        else:
            trace.completed, trace.completed_at_s, trace.result = True, scene.now(), task.result()
    else:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
    problems = [] if trace.error else kind.score(trace, expect)  # type: ignore[attr-defined]
    return CaseResult(step.id, case, trace, problems, time.monotonic() - began)


async def run_files(
    files: Sequence[Path],
    build: Build,
    recording: Recording,
    *,
    fakes: Any = None,
    only: Sequence[str] = (),
    concurrency: int = 4,
) -> list[CaseResult]:
    """Run every case of ``files`` (``only``: step ids or case names), ``concurrency`` at a time."""
    gate = asyncio.Semaphore(concurrency)
    jobs = []
    for path in files:
        spec = CaseFile.load(path)
        for case in spec.cases:
            if only and spec.step not in only and case.name not in only:
                continue
            step = build({**spec.build, **case.build}, fakes)[spec.step]

            async def one(step: Step[Any] = step, case: Case = case) -> CaseResult:
                async with gate:
                    return await run_case(step, case, recording)

            jobs.append(one())
    return list(await asyncio.gather(*jobs))


def report(results: Sequence[CaseResult]) -> str:
    """The results as Markdown: totals, then one table per step."""
    passed = sum(r.ok for r in results)
    lines = ["# Step cases", "", f"**{passed} of {len(results)} cases passed.**", ""]
    by_step: dict[str, list[CaseResult]] = {}
    for r in results:
        by_step.setdefault(r.step, []).append(r)
    for step, rows in by_step.items():
        lines += [f"## {step}", "", "| case | ok | completed | said | problems |", "|---|---|---|---|---|"]
        for r in rows:
            t = r.trace
            done = f"at {t.completed_at_s:.0f} s" if t.completed and t.completed_at_s is not None else "no"
            said = "; ".join(f"{x.kind} {x.t_s:.0f} s: {x.text}" for x in t.lines) or ""
            wrong = "; ".join(r.problems) or (t.error or "")
            lines.append(f"| {r.case.name} | {'yes' if r.ok else '**no**'} | {done} | {said} | {wrong} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None, *, build: Build, cases: Path, fakes_from: Callable[[Path], Any]) -> int:
    """The command line an app's evaluation script calls; the exit code is 0 when every case passed.

    ``cases`` is the folder of case files (one per step); ``fakes_from`` loads scripted roles for ``--fakes``.
    """
    parser = argparse.ArgumentParser(description="Run step cases on a recording.")
    parser.add_argument("--frames-index", type=Path, required=True, help="the recording's frame index CSV")
    parser.add_argument("--only", nargs="*", default=[], help="step ids or case names")
    parser.add_argument("--out", type=Path, default=Path("runs/steps"))
    parser.add_argument("--fakes", type=Path, help="scripted roles (a plumbing run; the numbers mean nothing)")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    from xr_ai_app.devkit import run_virtual  # noqa: PLC0415 - only when evaluating

    files = sorted(cases.glob("*.yaml"))
    fakes = fakes_from(args.fakes) if args.fakes else None
    recording = Recording.load(args.frames_index)
    results = run_virtual(run_files(files, build, recording, fakes=fakes, only=args.only, concurrency=args.concurrency))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps([r.as_json() for r in results], indent=2), encoding="utf-8")
    text = report(results)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0 if all(r.ok for r in results) else 1


def _summary(state: Any) -> str:
    summary = getattr(state, "summary", None)
    return summary() if callable(summary) else str(state)


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 2)
