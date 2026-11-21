# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Play a :class:`~xr_ai_app.devkit.Scenario` on a :class:`~xr_ai_app.devkit.LocalDevice` and write a report.

The report directory gets ``report.json`` (every step, every check, every input and output, pass or fail),
``transcript.md`` (what was said, by whom, when) and ``trace.jsonl`` (every trace event). A run passes when every
``wait_for``/``expect`` check found its text in time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._base.app import App
from .._base.testing import FakeModels
from .device import LocalDevice
from .scenario import Scenario, Step

__all__ = ["Report", "StepResult", "run_scenario"]


@dataclass
class StepResult:
    index: int
    action: str
    value: Any
    t: float
    ok: bool = True
    detail: str = ""


@dataclass
class Report:
    scenario: str
    passed: bool
    steps: list[StepResult] = field(default_factory=list)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    performance: dict[str, Any] | None = None

    @property
    def failures(self) -> list[StepResult]:
        return [s for s in self.steps if not s.ok]

    def to_json(self) -> str:
        return json.dumps(
            {
                "scenario": self.scenario,
                "passed": self.passed,
                "steps": [vars(s) for s in self.steps],
                "inputs": self.inputs,
                "outputs": self.outputs,
                **({"performance": self.performance} if self.performance is not None else {}),
            },
            indent=2,
            default=str,
            ensure_ascii=False,
        )

    def transcript(self) -> str:
        lines = [f"# {self.scenario}: {'PASSED' if self.passed else 'FAILED'}", ""]
        events = [(i["t"], 0, i) for i in self.inputs] + [(o["t"], 1, o) for o in self.outputs]
        for t, kind, item in sorted(events, key=lambda e: (e[0], e[1])):
            who = item["participant"]
            if kind == 1:
                lines.append(
                    f"- `{t:7.2f}s` **assistant -> {who}**{' (urgent)' if item['urgent'] else ''}: {item['text']}"
                )
            elif item["kind"] == "say":
                lines.append(f"- `{t:7.2f}s` **{who}**: {item['text']}")
            else:
                label = item.get("label", "")
                lines.append(f"- `{t:7.2f}s` _{who} {item['kind']}{': ' + label if label else ''}_")
        if self.failures:
            lines += ["", "## Failed checks", ""]
            lines += [f"- step {s.index} `{s.action}: {s.value}`: {s.detail}" for s in self.failures]
        return "\n".join(lines) + "\n"


async def run_scenario(
    app: App,
    scenario: Scenario,
    *,
    models: FakeModels | str | None = None,
    speed: float = 1.0,
    clock: str = "fast",
    out: str | Path | None = None,
    relay: bool = False,
    media_dir: str | Path | None = None,
    real_audio: bool = False,
    drain_s: float = 10.0,
    ready_timeout_s: float = 600.0,
    metadata: dict[str, Any] | None = None,
    images: dict[str, Path] | None = None,
) -> Report:
    """Play ``scenario`` against ``app``.

    ``clock="fast"`` skips the idle time between steps (media still plays at ``speed``); ``"real"`` waits for each
    step's ``at``. ``models`` as in :class:`LocalDevice`. ``out``: the report directory (nothing written when
    ``None``). ``relay``: also write the run through NeMo Relay as ``<out>/atof/`` (raw events) and
    ``<out>/atif/`` (a trajectory), the formats NeMo Compass and Helix read. ``media_dir``: look for each media
    path there first (your own recordings, under the same names), then next to the scenario.

    ``real_audio=True`` sends WAV steps through the app's real voice pipeline and an isolated hub,
    with a separate wall-clock sender. Requires ``models="real"``, ``clock="real"`` and ``speed=1``.
    ``drain_s`` adds response time after the fixed timeline; ``ready_timeout_s`` bounds startup.
    Its report includes ``performance``; ``metadata`` describes caller-owned hardware/services.
    """
    if clock not in ("fast", "real"):
        raise ValueError('clock is "fast" or "real"')
    if relay and out is None:
        raise ValueError("relay output needs an out directory")
    if real_audio:
        if models != "real" or clock != "real" or speed != 1:
            raise ValueError("real_audio requires models='real', clock='real' and speed=1")
        from ._audio import _prepare_audio, _run_audio  # noqa: PLC0415 - hub service is an optional dependency
        from .virtual_time import VirtualTimeLoop  # noqa: PLC0415

        if isinstance(asyncio.get_running_loop(), VirtualTimeLoop):
            raise ValueError("real_audio requires an actual wall-clock event loop")
        if out is not None and Path(out).exists():
            raise ValueError("real_audio output directory must not already exist")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        json.dumps(metadata)
        plan = _prepare_audio(scenario, media_dir, drain_s, images=images)
    elif images:
        raise ValueError("image bindings require real_audio=True; use media_dir for local-device scenarios")
    elif any(step.action == "audio" for step in scenario.steps):
        raise ValueError("audio steps require real_audio=True; use say for accuracy-only typed evaluation")
    if not real_audio and scenario.duration_s is not None:
        raise ValueError("duration_s requires real_audio=True")
    if real_audio:
        return await _run_audio(app, scenario, plan=plan, out=out, ready_timeout_s=ready_timeout_s,
                                metadata=metadata, relay=relay)
    async with contextlib.AsyncExitStack() as stack:
        if relay:
            await stack.enter_async_context(_relay_context(app, scenario, out))
        return await _run(app, scenario, models=models, speed=speed, clock=clock, out=out, media_dir=media_dir)


@contextlib.asynccontextmanager
async def _relay_context(app, scenario, out):
    async with contextlib.AsyncExitStack() as stack:
        from .._base._relay import RelaySink, relay_exporters  # noqa: PLC0415 - Relay only when asked

        session = "".join(c if c.isalnum() else "-" for c in scenario.name)[:64] or "run"
        await stack.enter_async_context(
            relay_exporters(
                atof_dir=Path(out) / "atof", atif_dir=Path(out) / "atif", session=session, agent=scenario.name
            )
        )
        sink = RelaySink()
        app.trace.add(sink)
        stack.callback(sink.close)  # runs after the removal below: no turn scope outlives the run
        stack.callback(app.trace.remove, sink)
        yield


async def _run(
    app: App,
    scenario: Scenario,
    *,
    models: FakeModels | str | None,
    speed: float,
    clock: str,
    out: str | Path | None,
    media_dir: str | Path | None = None,
) -> Report:
    results: list[StepResult] = []
    async with LocalDevice(app, models=models, speed=speed) as device:
        mark = 0  # outputs before this index came before the last user action
        previous_at = 0.0
        playing: dict[str, list[asyncio.Task[int]]] = {}  # background media, by participant
        for index, step in enumerate(scenario.steps):
            target = _target(step, previous_at)
            if clock == "real" and target is not None:
                delay = target - device.clock()
                if delay > 0:
                    await asyncio.sleep(delay)
            previous_at = target if target is not None else device.clock()
            result = StepResult(index, step.action, step.value, round(device.clock(), 3))
            participant = step.participant or scenario.participant
            try:
                media = _Media(scenario.base_dir, None if media_dir is None else Path(media_dir))
                mark = await _play(device, step, participant, media, mark, result, playing)
            except Exception as exc:  # noqa: BLE001 - a broken step fails the run, it does not crash it
                result.ok, result.detail = False, f"{type(exc).__name__}: {exc}"
            results.append(result)
        await _stop_media(playing, None)
        await device.settle()
        summary = device.summary()
        report = Report(scenario.name, all(r.ok for r in results), results, summary["inputs"], summary["outputs"])
        if out is not None:
            _write(report, device, Path(out))
    return report


def _target(step: Step, previous_at: float) -> float | None:
    if step.at is None:
        return None
    return previous_at + step.at if step.relative else step.at


@dataclass(frozen=True)
class _Media:
    """Where media paths resolve: ``media_dir`` first (the path as written, then its file name), then the scenario."""

    base: Path
    media_dir: Path | None

    def __truediv__(self, path: str) -> Path:
        if self.media_dir is not None:
            for candidate in (self.media_dir / path, self.media_dir / Path(path).name):
                if candidate.exists():
                    return candidate
        return self.base / path


async def _stop_media(playing: dict[str, list[asyncio.Task[int]]], participant: str | None) -> None:
    """Stop background media: one participant's (they left), or everyone's (the scenario ended)."""
    for who in [participant] if participant is not None else list(playing):
        for task in playing.pop(who, []):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def _media(
    clip: Coroutine[Any, Any, int],
    value: Any,
    participant: str,
    result: StepResult,
    playing: dict[str, list[asyncio.Task[int]]],
) -> None:
    """Play a clip: to its end before the next step, or (``background: true``) while the next steps run."""
    if not (isinstance(value, dict) and value.get("background")):
        result.detail = f"{await clip} frames"
        return
    task = asyncio.ensure_future(clip)
    playing.setdefault(participant, []).append(task)
    result.detail = "playing in the background"

    def finished(done: asyncio.Task[int]) -> None:
        if not done.cancelled() and done.exception() is None:
            result.detail = f"{done.result()} frames (background)"
        elif done.cancelled():
            result.detail = "stopped (the participant left or the scenario ended)"
        else:
            result.ok, result.detail = False, f"{type(done.exception()).__name__}: {done.exception()}"

    task.add_done_callback(finished)


async def _play(
    device: LocalDevice,
    step: Step,
    participant: str,
    base: _Media,
    mark: int,
    result: StepResult,
    playing: dict[str, list[asyncio.Task[int]]],
) -> int:
    value = step.value
    if step.action == "join":
        await device.join(str(value or participant))
    elif step.action == "leave":
        who = str(value or participant)
        await _stop_media(playing, who)
        await device.leave(who)
    elif step.action == "say":
        mark = len(device.hub.outputs)
        await device.say(str(value), participant)
        await device.settle()
    elif step.action == "image":
        if isinstance(value, dict) and value.get("stream", True) is False:
            await device.hold_image(base / str(value["path"]), participant)
        else:
            await device.show_image(base / str(value["path"] if isinstance(value, dict) else value), participant)
    elif step.action == "frames":
        clip = device.play_frames(base / value["dir"], fps=float(value.get("fps", 1.0)), participant=participant)
        await _media(clip, value, participant, result, playing)
    elif step.action == "video":
        clip = device.play_video(
            base / value["path"],
            fps=float(value.get("fps", 2.0)),
            from_s=float(value.get("from_s", 0.0)),
            to_s=None if value.get("to_s") is None else float(value["to_s"]),
            participant=participant,
        )
        await _media(clip, value, participant, result, playing)
    elif step.action == "sleep":
        await asyncio.sleep(float(value) / device.speed)
    elif step.action in ("wait_for", "expect"):
        spec = value if isinstance(value, dict) else {"said": value}
        needles = [str(n) for n in spec["said_any"]] if "said_any" in spec else [str(spec["said"])]
        found = await device.wait_for(
            needles,
            within_s=float(spec.get("within_s", 10.0)),
            after=mark,
            participant=spec.get("participant"),
        )
        if found is None:
            wanted = needles[0] if len(needles) == 1 else " or ".join(repr(n) for n in needles)
            result.ok, result.detail = False, f"nothing containing {wanted!r} within {spec.get('within_s', 10)}s"
        else:
            result.detail = f"at {found.t:.2f}s: {found.text}"
    return mark


def _write(report: Report, device: LocalDevice, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.to_json(), encoding="utf-8")
    (out / "transcript.md").write_text(report.transcript(), encoding="utf-8")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as file:
        for event in device.trace.all:
            file.write(event.to_json() + "\n")
