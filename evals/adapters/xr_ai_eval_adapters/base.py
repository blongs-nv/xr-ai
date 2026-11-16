# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared machinery: run a harness, timestamp its output, and turn parsed cases into records."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xr_ai_eval import AdapterContext, Record, load_profile
from xr_ai_eval.profiles import ProfileError

LATENCY_GAP = "gap_between_output_lines"
LATENCY_REPORTED = "harness_reported"
LATENCY_AVERAGE = "run_average"


@dataclass(frozen=True)
class Parsed:
    """One case result read from a harness, before it becomes a record."""

    case: str
    passed: bool
    score: float
    latency_ms: float | None = None
    latency_basis: str | None = None
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Line:
    """One harness output line with seconds since the harness started."""

    text: str
    at_s: float


@dataclass(frozen=True)
class HarnessRun:
    """What a harness process produced."""

    returncode: int
    lines: list[Line]
    elapsed_s: float


Executor = Callable[[list[str], Path, dict[str, str]], HarnessRun]


def run_streaming(argv: list[str], cwd: Path, env: dict[str, str]) -> HarnessRun:
    """Run ``argv`` and timestamp every line of combined stdout and stderr."""
    started = time.perf_counter()
    proc = subprocess.Popen(  # noqa: S603 -- argv is built by the adapter, never from user text
        argv,
        cwd=cwd,
        env={**os.environ, **env, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines = [Line(text.rstrip("\n"), time.perf_counter() - started) for text in proc.stdout]
    code = proc.wait()
    return HarnessRun(code, lines, time.perf_counter() - started)


@dataclass(frozen=True)
class Step:
    """One harness script. ``parse`` reads the run's lines and any files the script wrote."""

    prefix: str
    role: str
    script: str
    project: str
    parse: Callable[[HarnessRun, Path], list[Parsed]]
    uv_args: tuple[str, ...] = ()
    sample_models: str | None = None
    models_flag: str | None = None
    model_flag: str | None = None
    extra_args: Callable[[AdapterContext, Path], list[str]] = lambda ctx, work: []
    expected: Callable[[Path], list[str]] = lambda root: []


class HarnessAdapter:
    """Runs the steps of one existing harness and returns schema records."""

    def __init__(self, name: str, steps: list[Step], executor: Executor = run_streaming) -> None:
        self.name = name
        self.steps = steps
        self._executor = executor

    def with_executor(self, executor: Executor) -> HarnessAdapter:
        """A copy that runs processes through ``executor`` (tests inject recorded output)."""
        return HarnessAdapter(self.name, self.steps, executor)

    def command(self, step: Step, ctx: AdapterContext, work: Path) -> list[str]:
        """The argv for one step; flags are added only where the harness accepts them."""
        argv = ["uv", "--config-file", "uv.toml", "run", "--project", step.project, *step.uv_args]
        argv += ["python", *shlex.split(step.script)]
        if step.models_flag and ctx.profile_path is not None:
            argv += [step.models_flag, str(ctx.profile_path)]
            if step.model_flag and ctx.profile is not None:
                argv += [step.model_flag, ctx.profile.entry_key(step.role)]
        return [*argv, *step.extra_args(ctx, work)]

    def run(self, ctx: AdapterContext) -> list[Record]:
        """Run every step, convert each result to a record and mark cases that never reported."""
        records: list[Record] = []
        for step in self.steps:
            work = ctx.work_dir / step.prefix.replace("/", "_")
            work.mkdir(parents=True, exist_ok=True)
            result = self._executor(self.command(step, ctx, work), ctx.repo_root, {})
            records += self._convert(step, ctx, result, work)
        return records

    def _convert(self, step: Step, ctx: AdapterContext, result: HarnessRun, work: Path) -> list[Record]:
        parsed = step.parse(result, work)
        seen = {item.case for item in parsed}
        for name in step.expected(ctx.repo_root):
            if name not in seen:
                parsed.append(Parsed(name, False, 0.0, error="harness produced no result for this case"))
        if not parsed:
            tail = " | ".join(line.text for line in result.lines[-3:])
            error = f"harness exited {result.returncode} with no results: {tail}"
            parsed.append(Parsed(step.prefix or self.name, False, 0.0, error=error))
        honored = step.models_flag is not None and ctx.profile_path is not None
        preset, profile_name = self._preset(step, ctx, honored)
        average = result.elapsed_s * 1000.0 / len(parsed)
        records = []
        for item in parsed:
            detail = {
                **item.detail,
                "adapter": self.name,
                "harness_exit_code": result.returncode,
                "latency_basis": item.latency_basis if item.latency_ms is not None else LATENCY_AVERAGE,
                "profile_honored": honored,
                "requested_profile": ctx.profile_name,
            }
            records.append(
                Record(
                    suite=self.name,
                    case=f"{step.prefix}/{item.case}" if step.prefix else item.case,
                    role=step.role,
                    preset=preset,
                    profile=profile_name,
                    passed=item.passed and item.error is None,
                    score=item.score if item.error is None else 0.0,
                    latency_ms=round(item.latency_ms if item.latency_ms is not None else average, 3),
                    tokens={"input": None, "output": None},
                    gpu_profile=ctx.profile.gpu_profile if ctx.profile else None,
                    commit=ctx.commit,
                    run_id=ctx.run_id,
                    timestamp=ctx.timestamp,
                    error=item.error,
                    detail=detail,
                )
            )
        return records

    def _preset(self, step: Step, ctx: AdapterContext, honored: bool) -> tuple[str, str]:
        """Model and profile name to record: the profile's when the harness took it, else the sample's own."""
        if honored and ctx.profile is not None:
            try:
                return ctx.profile.endpoint(step.role).preset, ctx.profile.name
            except ProfileError:
                return "unknown", ctx.profile.name
        if step.sample_models:
            path = ctx.repo_root / step.sample_models
            try:
                return load_profile(str(path), []).endpoint(step.role).preset, "sample-models"
            except (ProfileError, OSError):
                pass
        return "unknown", "sample-models"


def gaps_ms(times_s: list[float]) -> list[float]:
    """Milliseconds from the previous timestamp (or the start) to each one."""
    out, previous = [], 0.0
    for at in times_s:
        out.append(max(0.0, (at - previous) * 1000.0))
        previous = at
    return out
