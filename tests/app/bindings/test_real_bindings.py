# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``App.from_config`` bindings, bound roles, the default voice agent and the idleness fallback."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from xr_ai_app._base import _bound
from xr_ai_app._base import app as app_module
from xr_ai_app._base._voice import build_voice_agent, models_for_voice
from xr_ai_app._base.app import App
from xr_ai_app._base.enrollment import PerConnection
from xr_ai_app._base.frames import Frame
from xr_ai_app._base.profiles import Role, load_profile
from xr_ai_models import DecisionAnswer, DecisionResponse, OpenAICompatLLM

_MODELS = {
    "models": {
        "llm": {"adapter": {"preset": "nemotron3_nano"}, "endpoint": {"base_url": "http://127.0.0.1:1"}},
        "vlm": {
            "category": "vlm",
            "adapter": {"kind": "openai_compat", "model_name": "v"},
            "endpoint": {"base_url": "http://127.0.0.1:2"},
        },
        "tts": {"adapter": {"preset": "pocket_tts"}, "endpoint": {"base_url": "http://127.0.0.1:3"}},
        "my_stt": {"adapter": {"preset": "parakeet_stt"}, "endpoint": {"base_url": "http://127.0.0.1:4"}},
    },
    "roles": {"speech_in": "my_stt", "answer": "llm", "perceive": "vlm", "speech_out": "tts"},
    "budgets": {"answer": {"max_concurrency": 1}},
}


def write_app(tmp_path: Path, models: dict[str, Any] | None = None, app_yaml: str = "{}") -> Path:
    (tmp_path / "models.json").write_text(json.dumps(models or _MODELS))
    path = tmp_path / "app.yaml"
    path.write_text(app_yaml)
    return path


class FakeChat:
    def __init__(self, reply: str = "hello") -> None:
        self.reply = reply
        self.messages: list[Any] = []
        self.closed = 0
        self.stream_closed = 0
        self.gate = asyncio.Event()
        self.in_flight = 0
        self.peak = 0

    async def chat(self, messages: Any, **_: Any) -> Any:
        self.messages.append(messages)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await self.gate.wait()
        finally:
            self.in_flight -= 1
        return SimpleNamespace(content=self.reply)

    async def ask_image(self, image: bytes, question: str, **_: Any) -> Any:
        self.messages.append((image, question))
        return SimpleNamespace(content=self.reply)

    def stream(self, *args: Any, **_: Any) -> AsyncIterator[str]:
        self.messages.append(args)
        return self._stream()

    async def _stream(self) -> AsyncIterator[str]:
        try:
            yield "a "
            await asyncio.sleep(10)
            yield "b"
        finally:
            self.stream_closed += 1

    async def close(self) -> None:
        self.closed += 1


@pytest.fixture
def fake_clients(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeChat]:
    made: dict[str, FakeChat] = {}

    def maker(_models: Any, entry: str) -> FakeChat:
        made[entry] = FakeChat()
        return made[entry]

    for category in ("llm", "vlm", "decision"):
        monkeypatch.setitem(_bound._MAKERS, category, maker)
    return made


class Answer(BaseModel):
    verdict: str


async def test_role_ask_binds_the_resolved_client_and_closes_it_on_stop(tmp_path, fake_clients) -> None:
    app = App.from_config(write_app(tmp_path), profile="models", enrollment=PerConnection(admit=None))
    role = app.role("answer")
    assert isinstance(role, Role) and role.name == "answer"
    assert app.role("answer") is role
    fake_clients["llm"].gate.set()
    assert await role.ask(None, "hi") == "hello"
    assert fake_clients["llm"].messages[0][0].content == "hi"
    with pytest.raises(KeyError, match="no model for role 'decide'"):
        app.role("decide")
    await app.start()
    await app.stop()
    assert fake_clients["llm"].closed == 1


async def test_role_ask_with_response_format_validates_and_vlm_needs_a_frame(tmp_path, fake_clients) -> None:
    app = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    fake_clients_llm = app.role("answer")
    fake_clients["llm"].gate.set()
    fake_clients["llm"].reply = '```json\n{"verdict": "yes"}\n```'
    assert await fake_clients_llm.ask(None, "q", response_format=Answer) == Answer(verdict="yes")
    assert "JSON" in fake_clients["llm"].messages[0][0].content
    fake_clients["llm"].reply = "not json"
    with pytest.raises(ValueError):
        await fake_clients_llm.ask(None, "q", response_format=Answer)
    frame = Frame(image=b"jpg", timestamp_us=1, width=1, height=1)
    with pytest.raises(ValueError, match="text model"):
        await fake_clients_llm.ask(frame, "q")
    vlm = app.role("perceive")
    assert await vlm.ask(frame, "see") == fake_clients["vlm"].reply
    assert fake_clients["vlm"].messages[0] == (b"jpg", "see")
    with pytest.raises(ValueError, match="needs a frame"):
        await vlm.ask(None, "see")
    await app.stop()


class Recording(FakeChat):
    """A chat/vision client that records the options each call carried."""

    def __init__(self, reply: str = "hello") -> None:
        super().__init__(reply)
        self.options: list[dict[str, Any]] = []
        self.gate.set()

    async def chat(self, messages: Any, **options: Any) -> Any:
        self.options.append(options)
        return await super().chat(messages)

    async def ask_image(self, image: bytes, question: str, **options: Any) -> Any:
        self.options.append(options)
        return await super().ask_image(image, question)

    async def ask_images(self, images: list[bytes], question: str, **options: Any) -> Any:
        self.options.append(options)
        self.messages.append((tuple(images), question))
        return SimpleNamespace(content=self.reply)

    def stream_images(self, images: list[bytes], question: str, **options: Any) -> AsyncIterator[str]:
        self.options.append(options)
        return self._stream()


async def test_ask_options_reach_the_client_and_system_is_a_separate_message(tmp_path, monkeypatch) -> None:
    made: dict[str, Recording] = {}

    def maker(_models: Any, entry: str) -> Recording:
        made[entry] = Recording()
        return made[entry]

    for category in ("llm", "vlm"):
        monkeypatch.setitem(_bound._MAKERS, category, maker)
    app = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    await app.role("answer").ask(
        None, "hi", system="be brief", thinking=False, max_tokens=7, temperature=0.1, timeout_s=5
    )
    roles = [m.role for m in made["llm"].messages[0]]
    assert roles == ["system", "user"] and made["llm"].messages[0][0].content == "be brief"
    assert made["llm"].options[0] == {"enable_thinking": False, "max_tokens": 7, "temperature": 0.1}
    await app.role("answer").ask(None, "plain")
    assert made["llm"].options[1] == {}  # nothing set, nothing sent
    f1 = Frame(image=b"a", timestamp_us=1, width=1, height=1)
    f2 = Frame(image=b"b", timestamp_us=2, width=1, height=1)
    vlm = app.role("perceive")
    await vlm.ask([f1, f2], "what changed?", system="compare", thinking=True, max_tokens=9)
    assert made["vlm"].messages[0] == ((b"a", b"b"), "what changed?")
    assert made["vlm"].options[0] == {"system_prompt": "compare", "max_tokens": 9}  # thinking has no switch here
    await vlm.ask([f1], "one")
    assert made["vlm"].messages[1] == (b"a", "one")
    with pytest.raises(ValueError, match="needs a frame"):
        await vlm.ask([], "none")
    with pytest.raises(ValueError, match="text model"):
        await app.role("answer").ask([f1], "no")
    stream = vlm.stream([f1, f2], "two")
    assert await stream.__anext__() == "a "
    await stream.aclose()
    await app.stop()


async def test_ask_timeout_raises_timeout_error_and_bad_structure_raises_structured_output_error(
    tmp_path, fake_clients
) -> None:
    from xr_ai_app._base.profiles import StructuredOutputError

    app = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    role = app.role("answer")  # the client's gate is closed: the call never returns
    with pytest.raises(TimeoutError):
        await role.ask(None, "slow", timeout_s=0.01)
    fake_clients["llm"].gate.set()
    fake_clients["llm"].reply = "nope"
    with pytest.raises(StructuredOutputError) as err:
        await role.ask(None, "q", response_format=Answer)
    assert isinstance(err.value, ValueError)
    with pytest.raises(TimeoutError):
        async for _ in role.stream(None, "talk", timeout_s=0.05):
            pass
    assert fake_clients["llm"].stream_closed == 1  # the timed-out stream was closed once


async def test_a_structured_reply_wrapped_in_the_schemas_properties_is_read(tmp_path, fake_clients) -> None:
    from xr_ai_app._base.profiles import StructuredOutputError

    app = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    role = app.role("answer")
    fake_clients["llm"].gate.set()
    fake_clients["llm"].reply = '{"properties": {"verdict": "yes"}}'  # Cosmos3-Nano (step 12)
    assert await role.ask(None, "q", response_format=Answer) == Answer(verdict="yes")
    fake_clients["llm"].reply = '{"title": "Answer", "properties": {"verdict": "no"}, "required": ["verdict"]}'
    assert await role.ask(None, "q", response_format=Answer) == Answer(verdict="no")
    fake_clients["llm"].reply = '{"properties": {"other": "yes"}}'
    with pytest.raises(StructuredOutputError):
        await role.ask(None, "q", response_format=Answer)
    await app.stop()


async def test_ask_cancellation_reaches_the_client_and_releases_the_budget(tmp_path, fake_clients) -> None:
    app = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    role = app.role("answer")
    client = fake_clients["llm"]
    tasks = [asyncio.ensure_future(role.ask(None, f"q{i}")) for i in range(3)]
    await asyncio.sleep(0.01)
    assert client.in_flight == 1  # max_concurrency: 1
    tasks[0].cancel()
    await asyncio.sleep(0.01)
    assert tasks[0].cancelled() and client.in_flight == 1  # the next waiter took the slot
    client.gate.set()
    assert [await t for t in tasks[1:]] == ["hello", "hello"]
    assert client.peak == 1


async def test_stream_cancel_closes_the_model_stream_exactly_once(tmp_path, fake_clients) -> None:
    app = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    role = app.role("answer")
    stream = role.stream(None, "talk")
    assert await stream.__anext__() == "a "

    async def pull() -> None:
        async for _ in stream:
            pass

    task = asyncio.ensure_future(pull())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()
    await stream.aclose()
    assert fake_clients["llm"].stream_closed == 1
    assert [x async for x in stream] == []


async def test_stream_closed_early_releases_the_concurrency_slot(tmp_path, fake_clients) -> None:
    app = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    role = app.role("answer")
    first = role.stream(None, "one")
    assert await first.__anext__() == "a "
    second = role.stream(None, "two")
    waiting = asyncio.ensure_future(second.__anext__())
    await asyncio.sleep(0.01)
    assert not waiting.done()
    await first.aclose()
    assert await asyncio.wait_for(waiting, 1) == "a "
    await second.aclose()


async def test_decide_uses_a_decision_model_and_rejects_text_roles(tmp_path, monkeypatch, fake_clients) -> None:
    models = json.loads(json.dumps(_MODELS))
    models["models"]["clef"] = {
        "category": "decision",
        "kind": "systemone",
        "model_name": "clef",
        "base_url": "http://127.0.0.1:5",
    }
    models["roles"]["decide"] = "clef"
    seen: dict[str, Any] = {}

    class Decision:
        async def decide(self, state: Any, questions: Any, **_: Any) -> DecisionResponse:
            seen.update(state=state, questions=questions)
            return DecisionResponse({"q": DecisionAnswer("b", 0.9, {"a": 0.1, "b": 0.9})}, "m", {}, {})

    monkeypatch.setitem(_bound._MAKERS, "decision", lambda *_: Decision())
    app = App.from_config(write_app(tmp_path, models), enrollment=PerConnection(admit=None))
    assert await app.role("decide").decide("which?", choices=["a", "b"], context="ctx") == "b"
    assert seen["state"] == "ctx" and set(seen["questions"]["q"].criteria) == {"a", "b"}
    with pytest.raises(ValueError, match="two distinct"):
        await app.role("decide").decide("which?", choices=["a"])
    with pytest.raises(TypeError, match="decision model"):
        await app.role("answer").decide("which?", choices=["a", "b"])


async def test_decide_maps_criteria_frames_and_reports_confidence(tmp_path, monkeypatch, fake_clients) -> None:
    from xr_ai_app._base.profiles import Decision

    models = json.loads(json.dumps(_MODELS))
    models["models"]["clef"] = {
        "category": "decision",
        "kind": "systemone",
        "model_name": "clef",
        "base_url": "http://127.0.0.1:5",
    }
    models["roles"]["decide"] = "clef"
    seen: dict[str, Any] = {}

    class Clef:
        async def decide(self, state: Any, questions: Any, **options: Any) -> DecisionResponse:
            seen.update(state=state, questions=questions, options=options)
            return DecisionResponse({"q": DecisionAnswer("b", 0.9, {"a": 0.1, "b": 0.9})}, "m", {}, {})

    monkeypatch.setitem(_bound._MAKERS, "decision", lambda *_: Clef())
    app = App.from_config(write_app(tmp_path, models), enrollment=PerConnection(admit=None))
    frame = Frame(image=b"jpg", timestamp_us=1, width=1, height=1)
    result = await app.role("decide").decide(
        "which?", choices={"a": "when A holds", "b": "when B holds"}, frames=[frame]
    )
    assert result == "b" and isinstance(result, Decision)
    assert result.confidence == 0.9 and result.probabilities == {"a": 0.1, "b": 0.9}
    assert seen["questions"]["q"].criteria == {"a": "when A holds", "b": "when B holds"}
    assert seen["options"] == {"images": [b"jpg"]}
    await app.role("decide").decide("which?", choices=["a", "b"])
    assert seen["options"] == {}  # text-only calls keep the text-only wire contract
    with pytest.raises(ValueError, match="criteria"):
        await app.role("decide").decide("which?", choices={"a": "x", "b": " "})


def test_role_without_a_profile_names_the_missing_file(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text("{}")
    with pytest.raises(FileNotFoundError, match="models"):
        App.from_config(path, enrollment=PerConnection(admit=None)).role("answer")


def test_real_clients_are_built_from_the_profile_without_connecting(tmp_path) -> None:
    app = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    assert isinstance(app._pool(app._load_profile()).client("llm"), OpenAICompatLLM)


def test_speech_in_is_exposed_to_the_voice_agent_as_the_stt_entry(tmp_path) -> None:
    profile = load_profile(write_app(tmp_path).parent / "models.json")
    models = models_for_voice(profile)
    assert models.entries["stt"] is profile.models.entries["my_stt"]
    assert "stt" not in profile.models.entries  # the profile itself is unchanged
    legacy = load_profile(write_app(tmp_path, {"models": {"stt": _MODELS["models"]["my_stt"]}}))
    assert models_for_voice(legacy) is legacy.models


async def test_default_voice_agent_publishes_no_presence(tmp_path, monkeypatch) -> None:
    """Presence has one publisher, the Hub agent (D-323a, R8.4); the voice pipeline is built with no presence topics."""
    import xr_ai_voice

    seen: dict[str, Any] = {}
    real = xr_ai_voice.VoiceAgent.from_models

    def capture(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(xr_ai_voice.VoiceAgent, "from_models", staticmethod(capture))
    app = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    voice = build_voice_agent(app, app._load_profile())
    assert seen["participant_joined_topic"] is None and seen["participant_left_topic"] is None
    assert voice.participant_joined_topic is None
    voice.request_stop()
    await voice._session.close()
    await app.stop()


async def test_default_voice_agent_never_installs_signal_handlers_and_uses_the_speech_roles(tmp_path) -> None:
    app = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    voice = build_voice_agent(app, app._load_profile())
    session = voice._session
    assert session.install_signal_handlers is False
    assert type(session.stt).__name__ == "OpenAICompatSTT"
    assert len(app._closeables) == 1  # the TTS client; the app closes it
    voice.request_stop()
    await session.close()
    await app.stop()


async def test_voice_options_default_to_the_current_behavior_and_reach_the_voice_agent(tmp_path) -> None:
    default = App.from_config(write_app(tmp_path), enrollment=PerConnection(admit=None))
    assert default._config.voice.idle_timeout_s is None and default._config.voice.interrupt_on_supersede is True
    voice = build_voice_agent(default, default._load_profile())
    assert voice._session.idle_timeout_secs is None and voice.interrupt_on_supersede is True
    voice.request_stop()
    await voice._session.close()
    await default.stop()

    other = tmp_path / "other"
    other.mkdir()
    custom = App.from_config(
        write_app(other, app_yaml="voice:\n  idle_timeout_s: 30\n  interrupt_on_supersede: false\n"),
        enrollment=PerConnection(admit=None),
    )
    assert custom._config.voice.idle_timeout_s == 30.0 and custom._config.voice.interrupt_on_supersede is False
    voice = build_voice_agent(custom, custom._load_profile())
    assert voice._session.idle_timeout_secs == 30.0 and voice.interrupt_on_supersede is False
    voice.request_stop()
    await voice._session.close()
    await custom.stop()


def test_voice_disabled_builds_no_voice_and_the_factory_is_installed_only_by_from_config(tmp_path) -> None:
    path = write_app(tmp_path, app_yaml="voice:\n  enabled: false\n")
    app = App.from_config(path, enrollment=PerConnection(admit=None))
    assert app._voice_factory is not None and app._voice_factory(app) is None
    from xr_ai_app._base.app import AppConfig

    assert App(AppConfig(config_dir=tmp_path))._voice_factory is None


async def test_from_config_app_start_stop_with_voice_built_by_the_factory(tmp_path, monkeypatch) -> None:
    app = App.from_config(
        write_app(tmp_path, app_yaml="timeouts: {drain_s: 0.5, hook_s: 0.5, close_s: 0.5}"),
        enrollment=PerConnection(admit=None),
    )
    events: list[str] = []

    class Voice:
        async def run(self, runtime: Any) -> None:
            events.append("run")
            await self.stopped.wait()

        def __init__(self) -> None:
            self.stopped = asyncio.Event()

        def request_stop(self) -> None:
            events.append("request_stop")
            self.stopped.set()

    monkeypatch.setattr("xr_ai_app._base._voice.build_voice_agent", lambda _app, _profile: Voice())
    await app.start()
    await asyncio.sleep(0.01)
    await app.stop()
    assert events == ["run", "request_stop"]


async def test_task_fut_waiter_exists_on_this_interpreter() -> None:
    """Fails loudly on a Python whose asyncio dropped the attribute ``App._is_idle`` reads."""
    task = asyncio.ensure_future(asyncio.sleep(10))
    await asyncio.sleep(0)
    assert hasattr(task, "_fut_waiter") and app_module._parked(task) is True
    task.cancel()


async def test_parked_falls_back_to_runnable_and_warns_once_without_fut_waiter(monkeypatch) -> None:
    class Bare:
        pass

    messages: list[str] = []
    monkeypatch.setattr(app_module, "_warned_no_waiter", False)
    sink = app_module.logger.add(messages.append, level="WARNING")
    try:
        assert app_module._parked(Bare()) is False  # type: ignore[arg-type]
        assert app_module._parked(Bare()) is False  # type: ignore[arg-type]
    finally:
        app_module.logger.remove(sink)
    assert len(messages) == 1 and "Python 3.11 to 3.14" in messages[0]


async def test_a_decision_role_bound_to_an_llm_entry_decides_and_is_probed_on_the_llm(tmp_path, monkeypatch) -> None:
    import httpx
    from xr_ai_app._base import _ready
    from xr_ai_models import ChatMessage, _openai_compat

    models = json.loads(json.dumps(_MODELS))
    models["models"]["router"] = {
        "category": "decision",
        "adapter": {"preset": "nemotron_omni"},
        "endpoint": {"base_url": "http://llm.test:8108"},
    }
    models["roles"]["decide"] = "router"
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": '{"q": "b"}'}}], "usage": {}}
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        _openai_compat.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw)
    )
    app = App.from_config(write_app(tmp_path, models), enrollment=PerConnection(admit=None))
    role = app.role("decide")
    frame = Frame(image=b"\x89PNG\r\n\x1a\nrest", timestamp_us=1, width=1, height=1)
    decision = await role.decide("which?", choices={"a": "when A holds", "b": "when B holds"}, frames=[frame])
    assert decision == "b" and decision.confidence == 1.0 and decision.probabilities == {"a": 0.0, "b": 1.0}
    (request,) = sent
    assert (request.url.host, request.url.path) == ("llm.test", "/v1/chat/completions")
    assert json.loads(request.read())["messages"][1]["content"][0]["type"] == "image_url"
    # The readiness wait probes the LLM endpoint; chat methods stay unavailable on a decision role.
    await _ready.wait_until_ready(app._probes)
    assert [r.url.path for r in sent[1:]] == ["/health"]
    with pytest.raises(TypeError, match="decision"):
        await role.ask(None, "hello")
    with pytest.raises(TypeError, match="decision"):
        await role.as_llm().chat([ChatMessage(role="user", content="hello")])
