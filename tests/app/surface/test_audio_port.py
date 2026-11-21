# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""App sample fixtures and invariants of the real-audio runner."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.surface._audio_fixtures import _app, _scenario, _wav
from PIL import Image
from xr_ai_app.devkit import Scenario, run_scenario
from xr_ai_app.devkit._audio import _prepare_audio
from xr_ai_app.devkit._fixtures import _prepare_fixtures
from xr_ai_hub import ProcessorEndpoint
from xr_ai_hub.isolation import isolated_endpoints
from xr_ai_tools.tool_calling import ToolLoopEmptyResponseError, observe_tool_errors

_ROOT = Path(__file__).resolve().parents[3]
_APPS = sorted((_ROOT / "app-samples").glob("**/eval/speech/*.yaml"))


@pytest.mark.integration
async def test_commands_without_replies_still_require_handler_delivery(tmp_path, monkeypatch):
    app, voice = _app(tmp_path, monkeypatch)
    raw = {
        "duration_s": 0.6,
        "steps": [{"join": "p1"}, {"at": 0.1, "audio": {"path": "speech.wav", "text": "capture", "reply": False}}],
    }
    _wav(tmp_path / "speech.wav")
    delivered = []

    @app.on_turn
    async def capture(turn, p):
        delivered.append(turn.text)
        await p.say("capture started", to=voice)

    result = await run_scenario(
        app,
        Scenario.from_dict(raw, base_dir=tmp_path),
        models="real",
        clock="real",
        real_audio=True,
        ready_timeout_s=5,
        drain_s=0.8,
    )
    assert all(step.ok for step in result.steps) and delivered
    assert result.performance["handler_coverage"] == {"received": 1, "expected": 1}
    assert result.performance["reply_coverage"] == {"completed": 0, "expected": 0}


def test_hub_scope_fails_closed_and_restores(tmp_path):
    args = {"sub_addr": f"ipc://{tmp_path}/pub", "push_addr": f"ipc://{tmp_path}/in"}
    with isolated_endpoints(**args):
        endpoint = ProcessorEndpoint("ipc:///tmp/xr_hub_pub", "ipc:///tmp/xr_hub_in")
        try:
            import zmq

            assert endpoint._sub.getsockopt_string(zmq.LAST_ENDPOINT) == args["sub_addr"]
            assert endpoint._push.getsockopt_string(zmq.LAST_ENDPOINT) == args["push_addr"]
        finally:
            endpoint.close()
        with pytest.raises(ValueError, match="private run hub"):
            ProcessorEndpoint("tcp://other:1234", "ipc:///tmp/xr_hub_in")
        with pytest.raises(ValueError, match="private run hub"):
            ProcessorEndpoint("ipc:///tmp/xr_hub_pub", "ipc:///tmp/xr_hub_in", file_sub_addr="ipc:///tmp/other")
    endpoint = ProcessorEndpoint(f"ipc://{tmp_path}/foreign", args["push_addr"])
    endpoint.close()


async def test_tool_error_observer_is_task_scoped_and_recovers():
    seen = []

    from xr_ai_models import ChatMessage, ChatResponse
    from xr_ai_tools.tool_calling import run_tool_loop

    async def empty(*_args):
        return ChatResponse(content="", reasoning=None, tool_calls=None, finish_reason="stop", raw={})

    async def fail():
        with pytest.raises(ToolLoopEmptyResponseError):
            await run_tool_loop([ChatMessage(role="user", content="hello")], tools={}, call_model=empty)

    with observe_tool_errors(seen.append):
        ToolLoopEmptyResponseError("constructed only", messages=[], tool_calls=[], iterations=1)
        assert seen == []
        await asyncio.create_task(fail())
        with observe_tool_errors(lambda _event: None):
            await fail()
        await fail()
    await fail()
    assert seen == [{"kind": "error", "exception_type": "ToolLoopEmptyResponseError"}] * 2


async def test_prepare_reuses_pinned_fixtures_without_tts_and_rejects_changes(tmp_path, monkeypatch):
    from xr_ai_app.devkit import _fixtures

    app, _voice = _app(tmp_path, monkeypatch)
    original = _scenario(tmp_path)
    scenario = dataclasses.replace(
        original,
        steps=tuple(
            dataclasses.replace(step, value={**step.value, "text": "hello"}) if step.action == "audio" else step
            for step in original.steps
        ),
    )
    output = tmp_path / "media"
    calls = []

    class TTS:
        async def synthesize(self, text, **_kwargs):
            calls.append(text)
            return _wav()

        async def close(self):
            pass

    monkeypatch.setattr(_fixtures, "make_tts", lambda *_args: TTS())
    image = tmp_path / "camera.png"
    path = await _prepare_fixtures(app, scenario, output, images={"camera.png": image})
    assert calls == ["hello"]
    assert json.loads(path.read_text())["files"]["speech.wav"] == hashlib.sha256(_wav()).hexdigest()
    await _prepare_fixtures(app, scenario, output, images={"camera.png": image})
    assert len(calls) == 1
    _prepare_audio(scenario, output, 0)
    (output / "speech.wav").write_bytes(_wav(seconds=0.2))
    with pytest.raises(ValueError, match="pinned fixture changed"):
        _prepare_audio(scenario, output, 0)
    with pytest.raises(ValueError, match="pinned fixture changed"):
        await _prepare_fixtures(app, scenario, output, images={"camera.png": image})


async def test_prepare_rejects_changed_words_and_missing_images_before_tts(tmp_path, monkeypatch):
    from xr_ai_app.devkit import _fixtures

    app, _voice = _app(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(_fixtures, "make_tts", lambda *_args: calls.append(1))
    with pytest.raises(ValueError, match="different utterances"):
        await _prepare_fixtures(app, _scenario(tmp_path), tmp_path / "media")
    scenario = Scenario.from_dict(
        {"steps": [{"join": "p1"}, {"image": "missing.png"}, {"audio": {"path": "a.wav", "text": "hello"}}]}
    )
    with pytest.raises(ValueError, match="missing image fixture"):
        await _prepare_fixtures(app, scenario, tmp_path / "missing-media")
    assert calls == []



@pytest.mark.parametrize("path", _APPS, ids=lambda path: str(path.relative_to(_ROOT)))
async def test_sample_owned_replays_are_preparable_without_services(path, tmp_path, monkeypatch):
    from xr_ai_app.devkit import _fixtures

    class TTS:
        async def synthesize(self, *_args, **_kwargs):
            return _wav()

        async def close(self):
            pass

    monkeypatch.setattr(_fixtures, "make_tts", lambda *_args: TTS())
    scenario = Scenario.load(path)
    target = SimpleNamespace(profile=SimpleNamespace(models=None), tts_entry="speech_out")

    async def stop():
        pass

    target.stop = stop
    monkeypatch.setattr(_fixtures, "_AppTarget", lambda _app: target)
    output = tmp_path / "prepared"
    await _prepare_fixtures(target, scenario, output)
    plan = _prepare_audio(scenario, output, 10)
    assert plan["duration_s"] == 300
    assert len(plan["fixture_sha256"]) >= 4


def test_image_aliases_reject_normalized_duplicates():
    from xr_ai_app.devkit.__main__ import _image_bindings

    with pytest.raises(ValueError, match="unique"):
        _image_bindings(["tea=a.png", "tea.png=b.png"])


async def test_prepare_selects_the_effective_component_tts_role(tmp_path, monkeypatch):
    from xr_ai_app.devkit import _fixtures

    app, voice = _app(tmp_path, monkeypatch)
    raw = json.loads((tmp_path / "models.json").read_text())
    raw["models"]["chosen-voice"] = raw["models"]["speech_out"]
    raw["roles"] = {"other-speech": "chosen-voice"}
    (tmp_path / "models.json").write_text(json.dumps(raw))
    voice._speech["tts"] = app.role("other-speech")
    scenario = Scenario.from_dict({"steps": [{"join": "p1"}, {"audio": {"path": "a.wav", "text": "hello"}}]})
    selected = []

    class TTS:
        async def synthesize(self, *_args, **_kwargs):
            return _wav()

        async def close(self):
            pass

    def make(models, entry):
        selected.append((models.entries[entry].endpoint.base_url, entry))
        return TTS()

    monkeypatch.setattr(_fixtures, "make_tts", make)
    await _prepare_fixtures(app, scenario, tmp_path / "prepared")
    assert selected == [("http://127.0.0.1:9", "chosen-voice")]


def test_camera_images_keep_original_hash_and_fit_shared_frame_limit(tmp_path):
    _wav(tmp_path / "speech.wav")
    image = tmp_path / "large.png"
    Image.new("RGB", (2048, 1536), "red").save(image)
    scenario = Scenario.from_dict(
        {"steps": [{"join": "p1"}, {"image": "large.png"}, {"audio": {"path": "speech.wav"}}]}, base_dir=tmp_path
    )
    plan = _prepare_audio(scenario, None, 0)
    frame = next(event for event in plan["events"] if event["action"] == "image")
    assert (frame["width"], frame["height"]) == (1280, 960)
    assert len(frame["pixels"]) < 6_000_000
    assert plan["fixture_sha256"]["large.png"] == hashlib.sha256(image.read_bytes()).hexdigest()


@pytest.mark.integration
async def test_changed_profile_cannot_be_reported_against_stale_app_clients(tmp_path, monkeypatch):
    app, _voice = _app(tmp_path, monkeypatch)
    app.role("speech_out")
    path = tmp_path / "models.json"
    raw = json.loads(path.read_text())
    raw["models"]["speech_out"]["endpoint"]["base_url"] = "http://different.example"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="profile changed"):
        await run_scenario(app, _scenario(tmp_path), models="real", clock="real", real_audio=True, out=tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.integration
async def test_caught_standard_logging_errors_are_measured_and_handler_is_restored(tmp_path, monkeypatch):
    import logging

    app, voice = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    before = list(logging.getLogger().handlers)

    @app.on_turn
    async def answer(turn, p):
        logging.getLogger("sample.recovered").error("sensitive message excluded from diagnostics")
        await p.reply("answer", to=voice)

    result = await run_scenario(
        app,
        _scenario(tmp_path),
        models="real",
        clock="real",
        real_audio=True,
        drain_s=0.5,
        ready_timeout_s=5,
        out=tmp_path / "standard-errors",
    )
    assert not result.passed
    assert result.performance["error_events"] == 2
    assert not result.performance["timing_comparable"]
    assert logging.getLogger().handlers == before
    assert "sensitive message" not in (tmp_path / "standard-errors/speech-events.jsonl").read_text()


def test_normal_background_cancellation_does_not_invalidate_completed_replay():
    from xr_ai_app.devkit._audio_report import _performance

    plan = {"events": [{"action": "audio", "index": 1, "handler": True}]}
    source = [
        {
            "kind": "input",
            "action": "audio",
            "index": 1,
            "handler": True,
            "text": "stop guidance",
            "monotonic": 10,
            "audio_start_us": 1000000,
            "audio_end_us": 1100000,
        },
        {"kind": "speech_end", "input_index": 1, "monotonic": 10.1},
        {"kind": "measurement_finished", "monotonic": 12, "max_sender_lateness_s": 0.001},
    ]
    observed = [
        (10.2, SimpleNamespace(kind="turn", turn="t1", data={"phase": "start", "timestamp_us": 1000000})),
        (10.3, SimpleNamespace(kind="response", turn="t1", data={"request_id": "t1", "final": True})),
        (10.35, SimpleNamespace(kind="turn", turn="t1", data={"phase": "end", "outcome": "done"})),
        (10.4, SimpleNamespace(kind="role", turn=None, data={"error": "CancelledError"})),
    ]
    result = _performance(
        plan, source, observed, [{"kind": "error", "monotonic": 10.5, "exception_type": "CancelledError"}], 10, 12, True
    )
    assert result["timing_comparable"] and result["error_events"] == 0
    observed.append((10.5, SimpleNamespace(kind="role", turn=None, data={"error": "ReadTimeout"})))
    result = _performance(plan, source, observed, [], 10, 12, True)
    assert not result["timing_comparable"] and result["error_events"] == 1


@pytest.mark.integration
async def test_new_capture_app_records_two_sessions_through_real_audio(tmp_path, monkeypatch):
    from xr_ai_app.devkit.device import load_app
    from xr_ai_app.testing import FakeModels
    from xr_ai_voice._processors import speaker_stt

    _app(tmp_path, monkeypatch)
    base = speaker_stt._SpeakerClient
    words = iter(["start capture", "first demonstration", "stop capture", "start capture", "stop capture"])

    class Speaker(base):
        async def _feed(self, *args):
            events = await super()._feed(*args)
            for event in events:
                if event["kind"] == "transcript":
                    event["text"] = next(words)
            return events

    monkeypatch.setattr(speaker_stt, "_SpeakerClient", Speaker)
    app = load_app(_ROOT / "app-samples/sdk-tour/12-procedure-capture/app.py")
    fake = FakeModels(
        {
            "perceive": [
                {
                    "caption": "A blue rectangle",
                    "delta": "Initial view",
                    "activity": "Demonstration",
                    "phase": "First step",
                }
            ]
            * 20
        }
    ).role("perceive")
    role = app._roles["perceive"]
    role.__class__ = type(fake)
    role.__dict__.clear()
    role.__dict__.update(fake.__dict__)
    app._probes.clear()
    _wav(tmp_path / "speech.wav")
    Image.new("RGB", (64, 48), "blue").save(tmp_path / "camera.png")
    steps = [{"join": "p1"}, {"image": "camera.png"}]
    for at, text in zip(
        [0.2, 1.2, 2.2, 3.2, 4.2],
        ["start capture", "first demonstration", "stop capture", "start capture", "stop capture"],
        strict=True,
    ):
        steps.append({"at": at, "audio": {"path": "speech.wav", "text": text, "reply": False}})
    result = await run_scenario(
        app,
        Scenario.from_dict({"duration_s": 5.2, "steps": steps}, base_dir=tmp_path),
        models="real",
        clock="real",
        real_audio=True,
        drain_s=0.5,
        ready_timeout_s=5,
        out=tmp_path / "capture",
    )
    assert result.passed, result.transcript()
    assert result.performance["handler_coverage"] == {"received": 5, "expected": 5}
    assert result.performance["timing_comparable"], result.performance
    packets = [json.loads(p.read_text()) for p in (tmp_path / "capture/artifacts").rglob("packet.json")]
    assert len(packets) == 2
    assert all(p["status"] == "complete" and p["stop_command"]["text"] == "stop capture" for p in packets)
    assert sum(p["counts"]["transcripts"] for p in packets) == 1
    assert all(p["counts"]["frames"] > 0 for p in packets)


def test_accuracy_cli_loads_credentials_before_hosted_app_profile(tmp_path, monkeypatch):
    import xr_ai_launcher
    from xr_ai_app.devkit import __main__ as cli
    from xr_ai_app.devkit.runner import Report

    calls = []
    monkeypatch.setattr(xr_ai_launcher, "load_credentials", lambda: calls.append("credentials"))

    def load(path, *, profile):
        assert calls == ["credentials"]
        calls.append((path, profile))
        return object()

    async def run(app, scenario, **kwargs):
        assert kwargs["models"] == "real" and not kwargs["real_audio"]
        return Report(scenario.name, True)

    monkeypatch.setattr(cli, "load_app", load)
    monkeypatch.setattr(cli, "run_scenario", run)
    scenario = tmp_path / "accuracy.yaml"
    scenario.write_text("steps: [{join: p1}, {say: hello}]\n")
    assert cli.main(["run", "app.py", str(scenario), "--profile", "hosted.json", "--real-models"]) == 0
    assert calls == ["credentials", ("app.py", "hosted.json")]


@pytest.mark.parametrize("previous", [None, "previous-profile"])
@pytest.mark.parametrize("fail", [False, True])
async def test_loader_binds_import_time_roles_to_profile_and_restores_environment(
    tmp_path, monkeypatch, previous, fail
):
    import os

    import httpx
    from xr_ai_app.devkit.device import load_app
    from xr_ai_models import _factory

    for name in ("original", "alternate"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "models": {
                        "answer": {
                            "category": "llm",
                            "adapter": {"kind": "openai_compat", "model_name": name},
                            "endpoint": {"base_url": f"https://{name}.example", "readiness": "none"},
                        }
                    }
                }
            )
        )
    (tmp_path / "app.yaml").write_text("profile: original.json\n")
    script = tmp_path / "app.py"
    script.write_text(
        "from pathlib import Path\nfrom xr_ai_app import App, PerConnection\n"
        "app = App.from_config(Path(__file__).with_name('app.yaml'), enrollment=PerConnection(admit=None))\n"
        + ("raise RuntimeError('injected import failure')\n" if fail else "app.role('answer')\n")
    )
    if previous is None:
        monkeypatch.delenv("XR_AI_PROFILE", raising=False)
    else:
        monkeypatch.setenv("XR_AI_PROFILE", previous)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "selected"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        factory = _factory.OpenAICompatLLM
        monkeypatch.setattr(_factory, "OpenAICompatLLM", lambda **kwargs: factory(client=client, **kwargs))
        if fail:
            with pytest.raises(RuntimeError, match="injected import failure"):
                load_app(script, profile=str(tmp_path / "alternate.json"))
        else:
            app = load_app(script, profile=str(tmp_path / "alternate.json"))
            try:
                assert await app.role("answer").ask(None, "test") == "selected"
                assert requests[0].url.host == "alternate.example"
                assert json.loads(requests[0].content)["model"] == "alternate"
            finally:
                await app.stop()
    assert os.environ.get("XR_AI_PROFILE") == previous


async def test_bad_tts_result_is_not_pinned_and_prepare_can_retry(tmp_path, monkeypatch):
    from xr_ai_app.devkit import _fixtures

    app, _voice = _app(tmp_path, monkeypatch)
    scenario = Scenario.from_dict(
        {"steps": [{"join": "p1"}, {"audio": {"path": "hello.wav", "text": "hello"}}]}, base_dir=tmp_path
    )
    media = tmp_path / "bad-tts"
    data = b"not a WAV"

    class TTS:
        async def synthesize(self, *_args, **_kwargs):
            return data

        async def close(self):
            pass

    monkeypatch.setattr(_fixtures, "make_tts", lambda *_args: TTS())
    import wave

    with pytest.raises(wave.Error):
        await _prepare_fixtures(app, scenario, media)
    assert not (media / "hello.wav").exists()
    assert not (media / "fixtures.json").exists()
    app, _voice = _app(tmp_path, monkeypatch)
    data = _wav()
    await _prepare_fixtures(app, scenario, media)
    assert json.loads((media / "fixtures.json").read_text())["utterances"] == {"hello.wav": "hello"}


def test_installed_source_does_not_claim_enclosing_repository(tmp_path, monkeypatch):
    from xr_ai_app.devkit import _targets

    monkeypatch.setattr(_targets, "_ROOT", tmp_path)
    monkeypatch.setattr(_targets.platform, "platform", lambda: "test-host")
    monkeypatch.setattr(_targets.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("git on installed SDK"))
    result = _targets._source_identity()
    assert result["revision"] is None and result["tracked_diff_sha256"] is None
    assert len(result["sdk_sha256"]) == 64


def test_accuracy_cli_works_without_optional_credential_loader(monkeypatch, capsys):
    import builtins

    from xr_ai_app.devkit import __main__ as cli

    original = builtins.__import__

    def without_launcher(name, *args, **kwargs):
        if name == "xr_ai_launcher":
            raise ModuleNotFoundError("optional launcher absent", name=name)
        return original(name, *args, **kwargs)

    calls = []
    monkeypatch.setattr(builtins, "__import__", without_launcher)
    monkeypatch.setattr(cli, "_run", lambda args: calls.append(args.real_models) or 0)
    assert cli.main(["run", "app.py", "scenario.yaml", "--real-models"]) == 0
    assert calls == [True]
    assert "environment credentials" in capsys.readouterr().err


@pytest.mark.parametrize("path", _APPS, ids=lambda path: str(path.relative_to(_ROOT)))
def test_speech_reuses_source_words_assertions_and_media(path):
    import yaml

    spec = yaml.safe_load(path.read_text())
    source = Scenario.load(path.parent / spec["scenario"])
    replay = Scenario.load(path)
    assert replay.base_dir == source.base_dir
    replay_images = [step.value for step in replay.steps if step.action == "image"]
    assert all(step.value in replay_images for step in source.steps if step.action == "image")
    assert [step.value["text"] for step in replay.steps if step.action == "audio" and step.value["handler"]] == [
        step.value for step in source.steps if step.action == "say"
    ]
    inherited = [step.value for step in source.steps if step.action in {"expect", "wait_for"}]
    checks = [{key: value for key, value in step.value.items() if key != "unprompted"}
              for step in replay.steps if step.action in {"expect", "wait_for"}]
    assert checks == inherited
    assert all((source.base_dir / (step.value["path"] if isinstance(step.value, dict) else step.value)).is_file()
               for step in replay.steps if step.action == "image")


async def test_source_scenario_edit_is_used_without_updating_speech_copy(tmp_path, monkeypatch):
    import yaml

    source = tmp_path / "base.yaml"
    speech = tmp_path / "speech.yaml"
    source.write_text("steps: [{join: p1}, {say: before}, {expect: {said: before}}, {leave: p1}]\n")
    speech.write_text(yaml.safe_dump({"scenario": "base.yaml", "schedule": [0, 1, 2, 5], "duration_s": 5}))
    first = Scenario.load(speech)
    source.write_text(source.read_text().replace("before", "after"))
    second = Scenario.load(speech)
    assert first.steps[1].value["text"] == "before"
    assert second.steps[1].value["text"] == "after"
    assert second.steps[2].value == {"said": "after"}
    assert first.steps[1].value["path"] != second.steps[1].value["path"]
    source.write_text(source.read_text().replace("{leave: p1}", "{say: extra}, {leave: p1}"))
    with pytest.raises(ValueError, match="one time for every"):
        Scenario.load(speech)


def test_tea_completion_deadline_contains_the_source_cases_steep_time(tmp_path):
    import yaml

    path = _ROOT / "app-samples/tea-making/eval/speech/realtime.yaml"
    scenario = Scenario.load(path)
    for step in scenario.steps:
        if step.action == "audio":
            _wav(tmp_path / step.value["path"])
    plan = _prepare_audio(scenario, tmp_path, 10)
    fakes = yaml.safe_load((scenario.base_dir / "guide.fakes.yaml").read_text())

    def steep_minutes(value):
        if isinstance(value, dict):
            if "minimum_minutes" in value:
                yield value["minimum_minutes"]
            for item in value.values():
                yield from steep_minutes(item)
        elif isinstance(value, list):
            for item in value:
                yield from steep_minutes(item)

    duration = next(steep_minutes(fakes)) * 60
    mug = next(step.at for step in scenario.steps if step.action == "image" and step.value["path"].endswith("mug.png"))
    final = next(check for check in plan["checks"] if check[2].get("said") == "Tea guidance is complete")
    assert final[3] <= mug + duration <= min(final[1] + final[5], plan["duration_s"] + plan["drain_s"])
