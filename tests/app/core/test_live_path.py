# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live path: speech in as upstream configures it, the on-demand picture, readiness, busy/idle.

Each test runs without a hub process or a model server: the voice agent is built but never run, endpoints are stubs,
and model clients are fakes with ``health``/``stream_images``.
"""

from __future__ import annotations

import asyncio
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from xr_ai_app import App, Hub, Participant, PerConnection, Turn, Voice
from xr_ai_app._base import _bound, _ready
from xr_ai_app._base._frames_feed import client_pictures, hub_endpoint
from xr_ai_app._base._voice import build_voice_agent
from xr_ai_app.testing import solid_frame
from xr_ai_hub._types import ImageCaptureData

REPO = Path(__file__).resolve().parents[3]
UPSTREAM = REPO / "agent-samples" / "simple-vlm-example" / "yaml"
SAMPLE = REPO / "app-samples" / "simple-vlm"


def _upstream_app(tmp_path: Path, app_yaml: str) -> Path:
    """An app directory holding upstream simple-vlm's models.json and voice_gate.yaml, unchanged."""
    (tmp_path / "yaml").mkdir()
    for name in ("models.json", "voice_gate.yaml"):
        shutil.copy(UPSTREAM / name, tmp_path / "yaml" / name)
    (tmp_path / "app.yaml").write_text(app_yaml)
    return tmp_path / "app.yaml"


async def _close(app: App, voice: Any) -> None:
    voice.request_stop()
    await voice._session.close()
    await app.stop()


def _same_but_the_recordings_folder(copy: Path, original: Path, folder: str) -> None:
    """``copy`` is upstream's ``media_capture.yaml`` with exactly one line changed: its own recordings folder."""
    text = original.read_text()
    old = next(line for line in text.splitlines() if line.startswith("out_dir: "))
    assert text.count(old) == 1
    assert copy.read_text() == text.replace(old, f"out_dir: ~/.local/share/xr-ai/captures/{folder}")
    assert copy.read_text() != text


# --- speech in and out from upstream's files -------------------------------------------------------------------------


def test_the_sample_carries_upstreams_model_and_gate_files_unchanged() -> None:
    for name in ("models.json", "voice_gate.yaml"):
        assert (SAMPLE / "yaml" / name).read_bytes() == (UPSTREAM / name).read_bytes()
    capture = "media_capture.yaml"
    _same_but_the_recordings_folder(SAMPLE / "yaml" / capture, UPSTREAM / capture, "xr-ai-app-simple-vlm")


def test_the_conversation_sample_carries_upstreams_files_and_prompts_unchanged() -> None:
    upstream = REPO / "agent-samples" / "conversation-example"
    sample = REPO / "app-samples" / "conversation"
    for name in ("models.json", "voice_gate.yaml"):
        assert (sample / "yaml" / name).read_bytes() == (upstream / "yaml" / name).read_bytes()
    capture = "yaml/media_capture.yaml"
    _same_but_the_recordings_folder(sample / capture, upstream / capture, "xr-ai-app-conversation")
    prompts = sorted((sample / "prompts").glob("*.txt"))
    assert len(prompts) == 7
    for prompt in prompts:
        original = upstream / "worker" / "conversation_example_worker" / "prompts" / prompt.name
        assert prompt.read_bytes() == original.read_bytes()


def test_the_lab_instruments_sample_carries_upstreams_files_and_prompts_unchanged() -> None:
    upstream = REPO / "agent-samples" / "lab-instrument-monitoring"
    sample = REPO / "app-samples" / "lab-instruments"
    for name in ("models.json", "voice_gate.yaml", "device_map.yaml"):
        assert (sample / "yaml" / name).read_bytes() == (upstream / "yaml" / name).read_bytes()
    capture = "yaml/media_capture.yaml"
    _same_but_the_recordings_folder(sample / capture, upstream / capture, "xr-ai-app-lab-instruments")
    prompts = sorted((sample / "prompts").glob("*.txt"))
    assert len(prompts) == 4
    for prompt in prompts:
        original = upstream / "worker" / "lab_instrument_monitoring_worker" / "prompts" / prompt.name
        assert prompt.read_bytes() == original.read_bytes()


TOUR = REPO / "app-samples" / "sdk-tour"


def test_the_sdk_tour_group_carries_upstream_conversations_hub_and_voice_files() -> None:
    upstream = REPO / "agent-samples" / "conversation-example" / "yaml"
    assert (TOUR / "yaml" / "voice_gate.yaml").read_bytes() == (upstream / "voice_gate.yaml").read_bytes()
    capture = "media_capture.yaml"
    _same_but_the_recordings_folder(TOUR / "yaml" / capture, upstream / capture, "xr-ai-app-sdk-tour")
    hub = (TOUR / "yaml" / "device_io_hub.yaml").read_text()
    assert hub == (upstream / "device_io_hub.yaml").read_text()  # the same depth as upstream's
    assert (TOUR / "yaml" / "../../../client-samples/web").resolve() == REPO / "client-samples" / "web"


def test_the_sdk_tour_launcher_runs_each_samples_app_with_its_own_project_and_the_shared_gate() -> None:
    import importlib.util  # noqa: PLC0415
    import tomllib  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("sdk_tour_launch", TOUR / "launch.py")
    assert spec is not None and spec.loader is not None
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    assert len(launch.SAMPLES) == 12
    assert launch.sample_dir("07") == launch.sample_dir("07-delegate-a2a/") == "07-delegate-a2a"
    assert launch.sample_dir("99") is None
    for sample in launch.SAMPLES:
        stack = launch.processes(sample)
        assert [p.name for p in stack][0] == "hub" and stack[-1].name == "app"
        app = stack[-1]
        scripts = tomllib.loads((TOUR / sample / "pyproject.toml").read_text())["project"]["scripts"]
        assert scripts == {app.command: "app:app.run"}  # the launcher's name for it, run by the sample's own uv
        assert (app.project, app.config) == (sample, f"{sample}/app.yaml")
        config = (TOUR / sample / "app.yaml").read_text()
        assert "profile: ../models.json" in config and "voice_gate_yaml: ../yaml/voice_gate.yaml" in config
    assert [p.name for p in launch.processes("07-delegate-a2a")] == ["hub", "research_agent", "app"]
    assert [p.name for p in launch.processes("01-hello-voice", capture=True)] == ["hub", "capture", "app"]


async def test_upstreams_files_give_required_speaker_asr_and_the_tts_entry(tmp_path) -> None:
    path = _upstream_app(tmp_path, "profile: yaml/models.json\nvoice: {voice_gate_yaml: yaml/voice_gate.yaml}\n")
    app = App.from_config(path, enrollment=PerConnection(admit=None))
    app.add(Voice(tts=app.role("tts"), text_input=False))
    voice = build_voice_agent(app, app._load_profile())
    session = voice._session
    speaker = session.voice_gate._speaker
    assert (speaker.backend, speaker.base_url) == ("required", "http://127.0.0.1:8102")
    assert session.voice_gate._mode == "conversation"
    assert session.stt is None  # speaker-only, as upstream's worker: no batch STT client
    assert session.tts is app._closeables[-1]  # the app closes it
    assert voice.text_input is False  # Voice(text_input=...) reaches the voice agent
    assert set(app._probes) == {"tts"}  # speech out is waited for before ready
    assert app._agent_status == session.transport.endpoint.set_status
    await _close(app, voice)


async def test_without_a_voice_component_speech_out_is_the_tts_entry(tmp_path) -> None:
    path = _upstream_app(tmp_path, "profile: yaml/models.json\nvoice: {voice_gate_yaml: yaml/voice_gate.yaml}\n")
    app = App.from_config(path, enrollment=PerConnection(admit=None))
    voice = build_voice_agent(app, app._load_profile())
    assert voice.text_input is True
    assert "tts" in app._probes
    await _close(app, voice)


async def test_a_speaker_speech_in_turns_speaker_asr_on_and_a_batch_one_leaves_the_file_in_force(tmp_path) -> None:
    models = json.loads((REPO / "app-samples" / "sdk-tour" / "models.json").read_text())
    (tmp_path / "models.json").write_text(json.dumps(models))
    (tmp_path / "app.yaml").write_text("{}")
    app = App.from_config(tmp_path / "app.yaml", enrollment=PerConnection(admit=None))
    voice = build_voice_agent(app, app._load_profile())  # the shared profile: speaker-stt on :8102
    assert voice._session.voice_gate._speaker.base_url == "http://127.0.0.1:8102"
    await _close(app, voice)

    batch = tmp_path / "batch"
    batch.mkdir()
    models["models"]["speech_in"] = {"adapter": {"preset": "parakeet_stt"}, "endpoint": {"base_url": "http://x:1"}}
    (batch / "models.json").write_text(json.dumps(models))
    shutil.copy(UPSTREAM / "voice_gate.yaml", batch / "voice_gate.yaml")  # speaker required
    (batch / "app.yaml").write_text("{}")
    app = App.from_config(batch / "app.yaml", enrollment=PerConnection(admit=None))
    voice = build_voice_agent(app, app._load_profile())
    assert (
        voice._session.voice_gate._speaker.backend == "required"
    )  # the file decides; the profile did not force it off
    await _close(app, voice)


async def test_voice_gate_from_the_component_wins_over_app_yaml(tmp_path) -> None:
    path = _upstream_app(tmp_path, "profile: yaml/models.json\n")  # no voice_gate_yaml: the default would be used
    app = App.from_config(path, enrollment=PerConnection(admit=None))
    app.add(Voice(gate="yaml/voice_gate.yaml"))
    voice = build_voice_agent(app, app._load_profile())
    assert voice._session.voice_gate._speaker.backend == "required"
    await _close(app, voice)


# --- the on-demand picture -------------------------------------------------------------------------------------------


def _png(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_the_hub_endpoint_subscribes_to_data_for_picture_replies() -> None:
    from xr_ai_hub import Subscribe

    endpoint = hub_endpoint()
    try:
        assert endpoint._default_filter == Subscribe.VIDEO | Subscribe.DATA
    finally:
        endpoint.close()


class CapturingEndpoint:
    """What upstream's capture source needs from a ``ProcessorEndpoint``; the client answers with ``reply``."""

    def __init__(self, reply: bytes | None, mime: str = "image/png") -> None:
        self.connected_participants = frozenset({"p1"})
        self.reply, self.mime = reply, mime
        self.image_cbs: list = []
        self.requests: list = []
        self.cancels: list = []

    def _on_image_capture(self, cb):  # noqa: ANN001, ANN202
        self.image_cbs.append(cb)
        return lambda: None

    def on_participant(self, cb) -> None:  # noqa: ANN001
        pass

    async def _request_image_capture(self, request) -> None:  # noqa: ANN001
        self.requests.append(request)
        if self.reply is not None:
            data = ImageCaptureData(request.participant_id, request.request_id, 1, self.mime, self.reply)
            asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(self.image_cbs[0](data)))

    async def _cancel_image_capture(self, cancel) -> None:  # noqa: ANN001
        self.cancels.append(cancel)


async def test_client_pictures_go_through_upstreams_capture_source() -> None:
    endpoint = CapturingEndpoint(_png())
    take = client_pictures(endpoint)
    frame = await take("p1")
    assert frame is not None and (frame.width, frame.height, frame.mime_type) == (8, 6, "image/png")
    assert frame.timestamp_us > 1  # an implausible client pts is replaced by the receive time
    assert [r.participant_id for r in endpoint.requests] == ["p1"]
    assert await take("nobody") is None  # not connected: no picture, no exception


async def test_a_cancelled_picture_cancels_the_request_on_the_client() -> None:
    endpoint = CapturingEndpoint(None)  # the client never answers
    take = client_pictures(endpoint)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(take("p1"), 0.05)
    assert len(endpoint.cancels) == 1


def _app() -> App:
    app = App.from_config(SAMPLE / "app.yaml", enrollment=PerConnection(admit=None))
    app.add(Hub())
    return app


async def test_current_takes_a_fresh_frame_first_then_a_picture_then_the_stream() -> None:
    app = _app()
    seen: list[Any] = []

    @app.on_turn
    async def answer(turn: Turn, p: Participant) -> None:
        seen.append(await p.frames.current(timeout_s=0.2))

    async with app.test_session() as s:
        await s.join()
        s.hub.hold_picture(solid_frame("blue"))
        await s.say("held")  # no stream: the held picture, asked for
        await s.settle()
        await s.show_frame(solid_frame("red"))
        await s.say("streamed")  # a fresh frame: no request
        await s.settle()
        events = [e.kind for e in s.hub.events]
        assert events.count("capture") == 1
        assert seen[0].image == solid_frame("blue").image and seen[1].image == solid_frame("red").image
        captures = s.trace.events(kind="frame")
        assert [e.data["ok"] for e in captures] == [True]


async def test_with_no_picture_current_waits_for_the_stream_as_before() -> None:
    app = _app()
    seen: list[Any] = []

    @app.on_turn
    async def answer(turn: Turn, p: Participant) -> None:
        seen.append(await p.frames.current(timeout_s=0.05))

    async with app.test_session() as s:
        await s.join()
        await s.say("anything there?")
        await s.settle()
        for _ in range(20):
            if seen:
                break
            await asyncio.sleep(0.02)
        assert seen == [None]
        assert [e.detail for e in s.hub.events if e.kind == "capture"] == [None]


# --- busy/idle --------------------------------------------------------------------------------------------------------


async def test_the_device_sees_processing_while_a_turn_runs_and_idle_after_also_when_superseded() -> None:
    app = _app()
    release = asyncio.Event()

    @app.on_turn
    async def answer(turn: Turn, p: Participant) -> None:
        if turn.text == "slow":
            await release.wait()

    async with app.test_session() as s:
        await s.join()
        await s.say("slow")
        await asyncio.sleep(0.02)
        assert s.hub.agent_status() == ["processing"]
        await s.say("quick")  # supersedes the slow turn
        await s.settle()
        assert s.hub.agent_status() == ["processing", "idle", "processing", "idle"]
        assert s.hub.statuses() == []  # error statuses are a separate record
        assert [e.name for e in s.trace.events(kind="status")] == ["processing", "idle", "processing", "idle"]


async def test_on_a_device_busy_idle_goes_to_the_voice_transports_endpoint() -> None:
    app = _app()
    sent: list[tuple[str, str]] = []

    async def set_status(status: str, participant_id: str) -> None:
        sent.append((status, participant_id))

    app._agent_status = set_status
    p = Participant("p1")
    await app._set_agent_status(p, "processing")
    assert sent == [("processing", "p1")]

    async def broken(status: str, participant_id: str) -> None:
        raise RuntimeError("endpoint closed")

    app._agent_status = broken
    await app._set_agent_status(p, "idle")  # contained: a lost status is not a failed turn


# --- readiness --------------------------------------------------------------------------------------------------------


class Client:
    def __init__(self, healthy: list[bool]) -> None:
        self.healthy = healthy
        self.asked: list[str] = []

    async def health(self) -> bool:
        return self.healthy.pop(0) if len(self.healthy) > 1 else self.healthy[0]

    async def stream_images(self, images, question, **_: Any):  # noqa: ANN001, ANN201
        self.asked.append(question)
        if not await self.health():
            raise ConnectionError("loading")
        yield "gray"

    async def close(self) -> None:
        pass


async def test_ready_only_once_every_bound_role_answers_and_a_vision_model_is_warmed(tmp_path, monkeypatch) -> None:
    llm, vlm = Client([False, True]), Client([False, False, True])
    made = {"vlm": vlm}
    monkeypatch.setitem(_bound._MAKERS, "vlm", lambda _models, entry: made[entry])
    monkeypatch.setitem(_bound._MAKERS, "llm", lambda _models, entry: llm)
    monkeypatch.setattr(_ready, "POLL_S", 0.01)
    (tmp_path / "models.json").write_text(
        json.dumps(
            {
                "models": {
                    "vlm": {
                        "category": "vlm",
                        "adapter": {"kind": "openai_compat", "model_name": "v"},
                        "endpoint": {"base_url": "http://127.0.0.1:2"},
                    },
                    "llm": {"adapter": {"preset": "nemotron3_nano"}, "endpoint": {"base_url": "http://127.0.0.1:1"}},
                }
            }
        )
    )
    (tmp_path / "app.yaml").write_text("voice: {enabled: false}\n")
    app = App.from_config(tmp_path / "app.yaml", enrollment=PerConnection(admit=None))
    app.role("vlm")
    app.role("llm")  # bound at import, as a sample does
    ready = tmp_path / "ready"
    stopped = asyncio.Event()

    async def run() -> None:
        # What App.run does with the launcher's --config and --ready-file: the profile and pool are rebuilt, the
        # roles bound at import keep their clients, and so do their probes.
        app._ready_file = ready
        app._roles.clear()
        app._profile_cache = app._pool_cache = None
        await app.start()
        await stopped.wait()

    task = asyncio.create_task(run())
    for _ in range(200):
        if ready.exists():
            break
        await asyncio.sleep(0.01)
    assert ready.exists()
    assert vlm.asked == ["What is the dominant color?"] * 3  # retried until the warm-up answered
    assert llm.healthy == [True]
    stopped.set()
    await task
    await app.stop()


async def test_stop_while_waiting_for_a_model_server_does_not_wait_out_the_drain(tmp_path, monkeypatch) -> None:
    down = Client([False])
    monkeypatch.setitem(_bound._MAKERS, "llm", lambda _models, entry: down)
    (tmp_path / "models.json").write_text(
        json.dumps({"models": {"llm": {"adapter": {"preset": "nemotron3_nano"}, "endpoint": {"base_url": "http://x"}}}})
    )
    (tmp_path / "app.yaml").write_text("voice: {enabled: false}\ntimeouts: {drain_s: 5}\n")
    app = App.from_config(tmp_path / "app.yaml", enrollment=PerConnection(admit=None))
    app.role("llm")
    app._ready_file = tmp_path / "ready"
    await app.start()
    await asyncio.sleep(0.05)
    loop = asyncio.get_running_loop()
    started = loop.time()
    await app.stop()
    assert loop.time() - started < 1.0
    assert not (tmp_path / "ready").exists()


async def test_every_takes_a_picture_per_tick_when_nothing_is_streamed() -> None:
    app = _app()
    seen: list[Any] = []

    async with app.test_session() as s:
        await s.join()
        s.hub.hold_picture(solid_frame("blue"))  # on-demand camera mode: nothing streamed
        p = s.participant()
        async for frame in p.frames.every(0.01):
            seen.append(frame)
            if len(seen) == 2:
                break
        assert [f.image for f in seen] == [solid_frame("blue").image] * 2
        assert [e.kind for e in s.hub.events].count("capture") == 2
        assert [e.data["ok"] for e in s.trace.events(kind="frame")] == [True, True]
