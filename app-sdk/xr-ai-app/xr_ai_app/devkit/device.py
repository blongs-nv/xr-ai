# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local device: drive any app the way a person with a headset would, without a hub.

:class:`LocalDevice` starts an app with an in-process device side (a recording ``FakeHub``) in place of the device
hub, and its roles either scripted (``FakeModels``) or real (the app's profile, only when asked for). It then joins
participants, types turns, shows images, holds a picture for the app to ask for (a camera in on-demand mode), plays
image folders and videos as timestamped frames, and records every output with its time. The scenario runner, the HTTP
endpoint and tests all use it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .._base._tracing import MemorySink
from .._base.app import App
from .._base.enrollment import PerConnection
from .._base.frames import Frame
from .._base.testing import FakeHub, FakeModels, ToolCall
from ..clock import wall_us

__all__ = ["LocalDevice", "Output", "fakes_from_file", "load_app"]

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _now() -> float:
    """The running loop's clock (a virtual-time loop moves it ahead), or ``time.monotonic()`` outside one."""
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        return time.monotonic()


@dataclass(frozen=True)
class Output:
    """One thing the app said: when (seconds since the device started), to whom, and how."""

    t: float
    participant: str
    text: str
    urgent: bool = False


class _RecordingHub(FakeHub):
    """A ``FakeHub`` that also timestamps every output, and tells ``on_capture`` about every picture handed over."""

    def __init__(self, clock: Callable[[], float], on_capture: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self._clock = clock
        self._on_capture = on_capture
        self.outputs: list[Output] = []
        self.changed = asyncio.Event()

    async def _take_picture(self, participant_id: str) -> Frame | None:
        frame = await super()._take_picture(participant_id)
        if frame is not None and self._on_capture is not None:
            self._on_capture(participant_id)
        return frame

    def publish_output(self, participant_id: str, text: str, *, urgent: bool = False) -> None:
        super().publish_output(participant_id, text, urgent=urgent)
        self.outputs.append(Output(round(self._clock(), 3), participant_id, text, urgent))
        self.changed.set()


def load_app(path: str | Path, attribute: str = "app", *, profile: str | None = None) -> App:
    """Import an app file (its directory goes on ``sys.path``, as when it runs) and return its ``app``."""
    file = Path(path).expanduser().resolve()
    if str(file.parent) not in sys.path:
        sys.path.insert(0, str(file.parent))
    spec = importlib.util.spec_from_file_location(f"xr_ai_devkit_app_{abs(hash(file))}", file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {file}")
    module = importlib.util.module_from_spec(spec)
    import os  # noqa: PLC0415

    previous = os.environ.get("XR_AI_PROFILE")
    try:
        if profile is not None:
            os.environ["XR_AI_PROFILE"] = profile
        spec.loader.exec_module(module)
    finally:
        if profile is not None:
            if previous is None:
                os.environ.pop("XR_AI_PROFILE", None)
            else:
                os.environ["XR_AI_PROFILE"] = previous
    app = getattr(module, attribute, None)
    if not isinstance(app, App):
        raise TypeError(f"{file}: no `{attribute}` App at module level")
    if profile is not None:
        app._profile = profile
    app._source_path = file
    return app


def fakes_from_file(path: str | Path) -> FakeModels:
    """Scripted roles from YAML.

    ``role: [reply, ...]`` answers in order. A reply is text, a mapping (a structured answer), or ``{tool_call: {name:
    ..., args: {...}}}``. When the order of a role's calls is not fixed (two components call it at once),
    answer by prompt instead::

        role:
          by_prompt:                       # the first entry whose text is in the call's prompt
            # (with its system text, and a decision's context: the request it judges)
            - ["List up to three", "Which cafes are near?"]
          otherwise: {tool_call: {name: start_research, args: {request: x}}}
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    script: dict[str, list[Any]] = {}
    for role, replies in raw.items():
        if isinstance(replies, dict) and "by_prompt" in replies:
            script[str(role)] = [_by_prompt(replies)] * 1000
        else:
            script[str(role)] = [_reply(reply) for reply in replies or []]
    return FakeModels(script)


def _reply(reply: Any) -> Any:
    if isinstance(reply, dict) and set(reply) == {"tool_call"}:
        call = reply["tool_call"]
        return ToolCall(call["name"], call.get("args") or {})
    return reply


def _by_prompt(spec: dict[str, Any]) -> Any:
    rules = [(str(needle), _reply(reply)) for needle, reply in spec["by_prompt"]]
    otherwise = _reply(spec["otherwise"]) if "otherwise" in spec else None

    def answer(call: Any) -> Any:
        text = f"{call.system or ''}\n{call.prompt or ''}\n{getattr(call, 'context', '') or ''}"
        for needle, reply in rules:
            if needle in text:
                return reply
        if otherwise is None:
            raise LookupError(f"no scripted reply matches the prompt: {(call.prompt or '')[:80]!r}")
        return otherwise

    return answer


def image_frame(data: bytes, *, timestamp_s: float | None = None) -> Frame:
    """An encoded image as a JPEG frame stamped now."""
    from PIL import Image  # noqa: PLC0415 - Pillow comes with the frame tools

    image = Image.open(io.BytesIO(data)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    stamp = wall_us() / 1_000_000 if timestamp_s is None else timestamp_s
    return Frame(buffer.getvalue(), int(stamp * 1_000_000), image.width, image.height)


class LocalDevice:
    """An app under local control (an async context manager).

    ``models`` is scripted roles, or ``"real"`` for the app's own profile (real model calls; only on request).
    ``speed`` scales media playback (2.0 plays a video twice as fast).
    """

    def __init__(self, app: App, *, models: FakeModels | str | None = None, speed: float = 1.0) -> None:
        if isinstance(models, str) and models != "real":
            raise ValueError('models is FakeModels, None (empty script) or "real"')
        self.app = app
        self.models = models if models is not None else FakeModels()
        self.speed = speed
        self._t0 = _now()
        self.hub = _RecordingHub(self.clock, on_capture=lambda who: self._input("capture", who))
        self.trace = MemorySink()
        self.inputs: list[dict[str, Any]] = []

    def clock(self) -> float:
        """Seconds since the device started."""
        return _now() - self._t0

    async def __aenter__(self) -> LocalDevice:
        if self.models == "real":
            self._install_hub_only()
        else:
            self.app._install_test_doubles(self.hub, self.models)  # type: ignore[arg-type]
        self.app.trace.add(self.trace)
        self._t0 = _now()
        await self.app.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            await self.app.stop()
        finally:
            self.hub.close()
            if self.trace in self.app.trace._sinks:
                self.app.trace.remove(self.trace)

    def _install_hub_only(self) -> None:
        """Like the test seam, but the roles stay the profile's real ones."""
        from .._base.hub import Hub  # noqa: PLC0415

        app = self.app
        app._hub = self.hub
        app._voice_factory = None
        if app._enrollment is None:
            app._enrollment = PerConnection(admit=None)
        app._swap_component(Hub, self.hub, add_if_missing=True)

    # ------------------------------------------------------------------ what a person does

    async def join(self, participant: str = "p1") -> None:
        self._input("join", participant)
        await self.hub.join(participant)
        await self.settle()

    async def leave(self, participant: str = "p1") -> None:
        self._input("leave", participant)
        await self.hub.leave(participant)
        await self.settle()

    async def say(self, text: str, participant: str = "p1") -> None:
        """Type a turn (speech to text is bypassed)."""
        self._input("say", participant, text=text)
        await self.hub.text_query(text, participant)

    async def show(self, frame: Frame, participant: str = "p1", *, label: str = "frame") -> None:
        self._input("frame", participant, label=label)
        await self.hub.push_frame(frame, participant)

    async def show_image(self, path: str | Path, participant: str = "p1") -> None:
        await self.show(image_frame(Path(path).read_bytes()), participant, label=str(path))

    async def hold_image(self, path: str | Path, participant: str = "p1") -> None:
        """Point the camera at an image without streaming it: the app gets it only by asking for a picture."""
        self._input("picture", participant, label=str(path))
        self.hub.hold_picture(image_frame(Path(path).read_bytes()), participant)

    async def play_frames(self, directory: str | Path, *, fps: float = 1.0, participant: str = "p1") -> int:
        """Show every image in ``directory`` (sorted by name) at ``fps``; returns how many."""
        files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        for index, file in enumerate(files):
            if index:
                await asyncio.sleep(1.0 / fps / self.speed)
            await self.show_image(file, participant)
        return len(files)

    async def play_video(
        self,
        path: str | Path,
        *,
        fps: float = 2.0,
        from_s: float = 0.0,
        to_s: float | None = None,
        participant: str = "p1",
    ) -> int:
        """Show a video's frames sampled at ``fps`` between ``from_s`` and ``to_s``; returns how many."""
        try:
            import cv2  # noqa: PLC0415 - optional: only video playback needs OpenCV
        except ImportError as exc:
            raise RuntimeError("playing a video needs OpenCV: depend on xr-ai-app[video]") from exc

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        shown = 0
        try:
            position = from_s
            while to_s is None or position <= to_s:
                capture.set(cv2.CAP_PROP_POS_MSEC, position * 1000.0)
                ok, image = capture.read()
                if not ok:
                    break
                ok, jpeg = cv2.imencode(".jpg", image)
                if not ok:
                    break
                if shown:
                    await asyncio.sleep(1.0 / fps / self.speed)
                height, width = image.shape[:2]
                frame = Frame(jpeg.tobytes(), wall_us(), width, height)
                await self.show(frame, participant, label=f"{Path(path).name}@{position:.2f}s")
                shown += 1
                position += 1.0 / fps
        finally:
            capture.release()
        return shown

    # ------------------------------------------------------------------ what the app did

    async def settle(self) -> None:
        """Wait until no fake model call is in flight and the app is idle (background loops may still run)."""
        from .._base.testing import TestSession  # noqa: PLC0415

        session = TestSession(self.app, self.hub, self.models if isinstance(self.models, FakeModels) else FakeModels())
        try:
            await session.settle()
        except TimeoutError:
            pass  # a long-running background task; callers wait on outputs instead

    def outputs(self, *, after: int = 0, participant: str | None = None) -> list[Output]:
        return [o for o in self.hub.outputs[after:] if participant is None or o.participant == participant]

    async def wait_for(
        self, said: str | Sequence[str], *, within_s: float = 10.0, after: int = 0, participant: str | None = None
    ) -> Output | None:
        """The first output from index ``after`` on containing ``said`` (or any of several; case-insensitive), or
        ``None`` in time."""
        deadline = _now() + within_s
        needles = [said.lower()] if isinstance(said, str) else [n.lower() for n in said]
        while True:
            for output in self.outputs(after=after, participant=participant):
                if any(needle in output.text.lower() for needle in needles):
                    return output
            remaining = deadline - _now()
            if remaining <= 0:
                return None
            self.hub.changed.clear()
            try:
                await asyncio.wait_for(self.hub.changed.wait(), timeout=min(remaining, 0.25))
            except TimeoutError:
                pass

    def _input(self, kind: str, participant: str, **data: Any) -> None:
        self.inputs.append({"t": round(self.clock(), 3), "kind": kind, "participant": participant, **data})

    def summary(self) -> Mapping[str, Any]:
        return {"inputs": self.inputs, "outputs": [asdict(o) for o in self.hub.outputs]}
