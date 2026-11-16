# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise one piece of an app on a clip: its real code, the app unchanged.

A **unit** is ``async unit(p, probe)``: it runs one piece of the app for participant ``p``, with ``probe`` for
what a harness adds (``probe.report(text)`` records a line with the video time, ``probe.turn(text)`` makes a
:class:`~xr_ai_app.Turn`, ``probe.video_time()`` is where the clip is). :func:`exercise` loads nothing new: it
starts the app on a :class:`~xr_ai_app.devkit.LocalDevice` (scripted roles, or the profile's real ones), joins
``p``, runs ``setup(p, probe)`` (inject context: fill a channel, set state), plays a window of a video (or a
folder of images) into ``p``'s camera, types any timed turns, and runs the unit until it returns, raises, or the
window plus ``linger_s`` runs out.

Units come from **targets**, named the way the app already names its pieces:

=========================  ======================================================================================
``fn:<module>:<attr>``     any ``async fn(p, probe)``, imported from a file next to the app or a module
``destination:<name>``     a router destination run on one turn: ``options={"text": ...}`` (``xr-ai-kit``
                           registers it)
``step:<step id>``         a procedure step, run on its own (``xr-ai-procedures`` registers it)
=========================  ======================================================================================

A package adds a kind with :func:`target_kind` or the ``xr_ai_app.exercise_targets`` entry-point group
(``kind = "module:resolver"``, ``resolver(app, name, options) -> unit``); a resolver finds the app's pieces with
:func:`components`.

The :class:`Exercise` it returns has the unit's result (or error), the video time it ended at, the probe's
reports, what the app said, and the trace events of the run (the model calls with their answers).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from .._base.component import Component
from .._base.participant import Turn
from ..clock import now, wall_us

if TYPE_CHECKING:
    from .._base.app import App
    from .._base.participant import Participant
    from .._base.testing import FakeModels
    from .device import LocalDevice

__all__ = ["Exercise", "Probe", "Unit", "components", "exercise", "resolve_target", "target_kind"]

C = TypeVar("C", bound=Component)

Unit = Callable[["Participant", "Probe"], Awaitable[Any]]
Resolver = Callable[["App", str, Mapping[str, Any]], Unit]

_KINDS: dict[str, Resolver] = {}


def target_kind(kind: str) -> Callable[[Resolver], Resolver]:
    """Register ``resolver(app, name, options) -> unit`` for targets ``<kind>:<name>``."""

    def register(resolver: Resolver) -> Resolver:
        _KINDS[kind] = resolver
        return resolver

    return register


class Probe:
    """What a unit gets besides ``p``: where the clip is, a way to report, turns, and when the clip ended.

    A unit that would run on after its clip (a whole procedure) waits for :attr:`clip_ended` and returns what it has
    within :attr:`linger_s`; otherwise it is cancelled and its result is lost.
    """

    def __init__(
        self,
        device: LocalDevice,
        participant: str,
        *,
        video: bool = False,
        from_s: float = 0.0,
        to_s: float | None = None,
        linger_s: float = 3.0,
    ) -> None:
        self._device = device
        self._participant = participant
        self._video, self._from_s, self._to_s = video, from_s, to_s
        self._started: float | None = None
        self.linger_s = linger_s
        self.clip_ended = asyncio.Event()
        self.reports: list[tuple[float | None, str]] = []
        self._turns = 0

    def _start(self) -> None:
        self._started = now()

    def video_time(self) -> float | None:
        """Where the clip is, in clip seconds: for a video, the window start plus the time since it started (capped
        at the window end); for images, ``None``."""
        if not self._video or self._started is None:
            return None
        at = self._from_s + (now() - self._started) * self._device.speed
        return round(at if self._to_s is None else min(at, self._to_s), 3)

    async def report(self, text: str) -> None:
        """Record one line of what the unit sees or does, at the current clip time."""
        self.reports.append((self.video_time(), text))

    def turn(self, text: str) -> Turn:
        """A user turn with this text, stamped now."""
        self._turns += 1
        return Turn(text=text, timestamp_us=_now_us(), id=f"exercise-{self._turns}")


@dataclass
class Exercise:
    """What one exercise did."""

    target: str
    status: Literal["returned", "raised", "unfinished"]
    result: Any = None
    error: str = ""
    ended_at_s: float | None = None
    """Clip time when the unit returned or raised (``None`` when unfinished or without video)."""
    frames_shown: int = 0
    reports: list[tuple[float | None, str]] = field(default_factory=list)
    said: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def outcome(self) -> str:
        """The result's type name (``Done``, ``Needs``, ``Stuck``, ``DestinationResult``...), or the status."""
        if self.status != "returned":
            return self.status
        return "None" if self.result is None else type(self.result).__name__

    @property
    def passed(self) -> bool:
        """It returned, and its result does not say it failed (a result with a ``passed`` attribute judges itself)."""
        return self.status == "returned" and bool(getattr(self.result, "passed", True))

    @property
    def calls(self) -> list[dict[str, Any]]:
        """The model calls made during the run (trace ``role`` events)."""
        return [e for e in self.events if e["kind"] == "role"]

    def to_json(self) -> str:
        return json.dumps(
            {
                "target": self.target,
                "status": self.status,
                "outcome": self.outcome,
                "result": _plain(self.result),
                "error": self.error,
                "ended_at_s": self.ended_at_s,
                "frames_shown": self.frames_shown,
                "reports": [{"at_s": t, "text": text} for t, text in self.reports],
                "said": self.said,
                "calls": self.calls,
            },
            indent=2,
            default=str,
            ensure_ascii=False,
        )

    def write(self, out: str | Path) -> None:
        """``exercise.json`` (the summary) and ``trace.jsonl`` (every event) under ``out``."""
        directory = Path(out)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "exercise.json").write_text(self.to_json() + "\n", encoding="utf-8")
        with (directory / "trace.jsonl").open("w", encoding="utf-8") as file:
            for event in self.events:
                file.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")


async def exercise(
    app: App,
    unit: Unit,
    *,
    target: str = "unit",
    video: str | Path | None = None,
    from_s: float = 0.0,
    to_s: float | None = None,
    frames_dir: str | Path | None = None,
    fps: float = 2.0,
    say: Sequence[tuple[float, str]] = (),
    setup: Callable[[Participant, Probe], Any] | None = None,
    models: FakeModels | str | None = None,
    speed: float = 1.0,
    participant: str = "p1",
    linger_s: float = 3.0,
    max_s: float = 600.0,
) -> Exercise:
    """Run ``unit`` on a window of ``video`` (or ``frames_dir``) and return what happened (module docstring).

    ``say`` is ``[(seconds after the start, text), ...]``: turns typed into the app while the unit runs (they go
    through the app's own turn handling). ``models`` as in :class:`LocalDevice` (``"real"`` only when asked). With a
    clip the unit may run ``linger_s`` past its end; without one it runs until it returns. Either way it is stopped
    after ``max_s``. On a :class:`~xr_ai_app.devkit.VirtualTimeLoop` these are virtual seconds.
    """
    from .device import LocalDevice  # noqa: PLC0415 - avoid a cycle with the devkit package

    async with LocalDevice(app, models=models, speed=speed) as device:
        await device.join(participant)
        await device.settle()
        p = app._participants.get(participant)
        if p is None:
            raise RuntimeError(f"participant {participant!r} did not join")
        probe = Probe(device, participant, video=video is not None, from_s=from_s, to_s=to_s, linger_s=linger_s)
        if setup is not None:
            value = setup(p, probe)
            if inspect.isawaitable(value):
                await value
        events_mark, said_mark = len(device.trace.all), len(device.hub.outputs)
        shown: list[int] = []

        async def feed() -> None:
            try:
                if video is not None:
                    shown.append(
                        await device.play_video(video, fps=fps, from_s=from_s, to_s=to_s, participant=participant)
                    )
                elif frames_dir is not None:
                    shown.append(await device.play_frames(frames_dir, fps=fps, participant=participant))
            finally:
                probe.clip_ended.set()

        async def talk() -> None:
            started = asyncio.get_running_loop().time()
            for at, text in sorted(say):
                delay = at / device.speed - (asyncio.get_running_loop().time() - started)
                if delay > 0:
                    await asyncio.sleep(delay)
                await device.say(text, participant)

        probe._start()
        inputs = asyncio.gather(feed(), talk())
        task = asyncio.ensure_future(unit(p, probe))
        has_clip = video is not None or frames_dir is not None or bool(say)
        await asyncio.wait({task, inputs} if has_clip else {task}, timeout=max_s, return_when=asyncio.FIRST_COMPLETED)
        if not task.done() and has_clip and inputs.done():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), linger_s)
        ended_at = probe.video_time()
        if inputs.done() and not inputs.cancelled() and inputs.exception() is not None:
            task.cancel()  # the clip could not be played: a harness error, not the unit's outcome
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise inputs.exception()  # type: ignore[misc]
        if not inputs.done():
            inputs.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await inputs
        result: Exercise
        if task.done() and not task.cancelled():
            exc = task.exception()
            status: Literal["returned", "raised", "unfinished"] = "raised" if exc else "returned"
            result = Exercise(
                target,
                status,
                task.result() if exc is None else None,
                f"{type(exc).__name__}: {exc}" if exc else "",
                ended_at,
            )
        else:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            result = Exercise(target, "unfinished")
        await device.settle()
        result.frames_shown = sum(shown)
        result.reports = list(probe.reports)
        result.said = [asdict(o) for o in device.hub.outputs[said_mark:]]
        result.events = [json.loads(e.to_json()) for e in device.trace.all[events_mark:]]
        return result


def resolve_target(
    app: App, target: str, options: Mapping[str, Any] | None = None, *, base: Path | None = None
) -> Unit:
    """The unit for ``<kind>:<name>`` (see the module docstring)."""
    kind, colon, name = target.partition(":")
    if not colon:
        raise ValueError(f"a target is <kind>:<name>, got {target!r}")
    options = dict(options or {})
    if kind == "fn":
        return _function(name, base)
    resolver = _KINDS.get(kind) or _entry_point(kind)
    if resolver is None:
        known = sorted({"fn", *_KINDS, *(ep.name for ep in entry_points(group=_GROUP))})
        raise ValueError(f"unknown target kind {kind!r}; known: {', '.join(known)}")
    return resolver(app, name, options)


_GROUP = "xr_ai_app.exercise_targets"


def _entry_point(kind: str) -> Resolver | None:
    for ep in entry_points(group=_GROUP):
        if ep.name == kind:
            resolver = ep.load()
            _KINDS[kind] = resolver
            return resolver
    return None


def components(app: App, kind: type[C]) -> list[C]:
    """The app's components that are instances of ``kind``, in the order they were added (a new list).

    For target resolvers: they find the piece a target names (a procedure, a router) on the app they are given.
    ``components(app, Component)`` is every component.
    """
    return [c for c in app._components if isinstance(c, kind)]


def _function(spec: str, base: Path | None) -> Unit:
    module_name, _, attr = spec.rpartition(":")
    if not module_name:
        raise ValueError(f"fn targets are fn:<module or file.py>:<attr>, got fn:{spec}")
    if module_name.endswith(".py"):
        path = Path(module_name)
        if not path.is_absolute() and base is not None:
            path = base / path
        loaded = importlib.util.spec_from_file_location(f"_exercise_{path.stem}", path)
        if loaded is None or loaded.loader is None:
            raise FileNotFoundError(path)
        module = importlib.util.module_from_spec(loaded)
        sys.modules[loaded.name] = module
        loaded.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__name__, **asdict(value)}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return {"type": type(value).__name__, **dump()}
    return value


def _now_us() -> int:
    return wall_us()
