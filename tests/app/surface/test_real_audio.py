# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real IPC and native voice-pipeline tests, with deterministic speech services on CPU."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from app.surface._audio_fixtures import _app, _scenario, _wav
from xr_ai_app import App
from xr_ai_app.devkit import Scenario, run_scenario
from xr_ai_app.devkit._audio import _prepare_audio


@pytest.mark.integration
@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("relay", [False, True])
async def test_real_audio_runs_native_stt_and_tts_and_keeps_source_independent(tmp_path, monkeypatch, block, relay):
    app, voice = _app(tmp_path, monkeypatch)
    scenario = _scenario(tmp_path)
    seen = []

    @app.on_turn
    async def answer(turn, p):
        seen.append(turn.timestamp_us)
        frame = await p.frames.current()
        assert frame is not None
        if block:
            time.sleep(0.35)
        await p.reply("answer", to=voice)

    output = tmp_path / "run"
    report = await run_scenario(
        app,
        scenario,
        models="real",
        clock="real",
        real_audio=True,
        drain_s=1,
        ready_timeout_s=5,
        out=output,
        relay=relay,
    )
    assert all(step.ok for step in report.steps), report.transcript()
    if relay:
        assert list((output / "atof").glob("*"))
        assert list((output / "atif").glob("*.json"))
    metrics = report.performance
    assert metrics["execution_status"] == "completed"
    assert metrics["handler_coverage"] == {"received": 2, "expected": 2}
    assert metrics["unattributed_turns"] == 0
    assert metrics["completed_reply_text_s"]["n"] == 2
    assert metrics["first_observed_audio_s"]["n"] >= 1
    if block:
        assert metrics["max_sender_lateness_s"] < 0.1
        assert metrics["timing_comparable"]
    assert metrics["stt"]["max_queue_s"] >= 0
    assert metrics["error_events"] == 0
    assert len(seen) == 2
    assert not app._participants
    assert app._ready_file is None
    saved = json.loads((output / "report.json").read_text())
    assert saved["performance"] == metrics
    rows = [json.loads(line) for line in (output / "audio-events.jsonl").read_text().splitlines()]
    assert any(row["kind"] == "audio_out" and row["rms"] > 0 for row in rows)
    assert (
        metrics["configuration"]["fixture_sha256"]["speech.wav"]
        == hashlib.sha256((tmp_path / "speech.wav").read_bytes()).hexdigest()
    )


@pytest.mark.integration
@pytest.mark.parametrize("explicit_voice", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
async def test_audio_report_records_default_and_explicit_voice_replies(tmp_path, monkeypatch, explicit_voice, streamed):
    app, voice = _app(tmp_path, monkeypatch)

    async def fragments():
        yield "ans"
        yield "wer"

    @app.on_turn
    async def answer(turn, p):
        await p.reply(fragments() if streamed else "answer", to=voice if explicit_voice else None)

    report = await run_scenario(
        app, _scenario(tmp_path), models="real", clock="real", real_audio=True, drain_s=1, ready_timeout_s=5
    )
    assert all(step.ok for step in report.steps), report.transcript()
    assert [output["text"] for output in report.outputs] == ["answer", "answer"]
    assert report.performance["reply_coverage"] == {"completed": 2, "expected": 2}


@pytest.mark.parametrize(
    "change, message",
    [
        ({"duration_s": -1}, "duration_s"),
        ({"steps": [{"join": "p1"}, {"audio": {"path": "absent.wav"}}]}, "absent.wav"),
        ({"steps": [{"join": "p1"}, {"audio": {"path": "speech.wav", "sha256": "bad"}}]}, "hash mismatch"),
        ({"steps": [{"join": "p1"}, {"audio": {"path": "speech.wav"}}, {"audio": {"path": "speech.wav"}}]}, "overlap"),
        ({"steps": [{"join": "p1"}, {"say": "hi"}]}, "say"),
    ],
)
def test_real_audio_preflight_rejects_bad_inputs(tmp_path, change, message):
    _wav(tmp_path / "speech.wav")
    raw = {"steps": [{"join": "p1"}, {"audio": {"path": "speech.wav"}}], **change}
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        _prepare_audio(Scenario.from_dict(raw, base_dir=tmp_path), None, 0)


@pytest.mark.integration
async def test_audio_requires_explicit_real_time_and_models(tmp_path):
    scenario = _scenario(tmp_path)
    from xr_ai_app._base.app import AppConfig

    app = App(AppConfig(config_dir=tmp_path))
    with pytest.raises(ValueError, match="requires models"):
        await run_scenario(app, scenario, real_audio=True)
    with pytest.raises(ValueError, match="audio steps require"):
        await run_scenario(app, scenario)


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["handler", "missing_transcript", "source"])
async def test_real_audio_failure_is_visible_and_owned_resources_are_released(tmp_path, monkeypatch, failure):
    from multiprocessing import shared_memory
    from types import SimpleNamespace

    from xr_ai_app.devkit import _audio

    app, voice = _app(tmp_path, monkeypatch, transcribe=failure != "missing_transcript")
    context = _audio.multiprocessing.get_context("spawn")
    owned = []

    def process(**kwargs):
        result = context.Process(**kwargs)
        owned.append((result, kwargs["args"][4].replace("-", "_")))
        return result

    monkeypatch.setattr(
        _audio.multiprocessing, "get_context", lambda _kind: SimpleNamespace(Pipe=context.Pipe, Process=process)
    )

    @app.on_turn
    async def answer(turn, p):
        if failure == "handler":
            raise RuntimeError("injected handler failure")
        if failure == "source":
            owned[0][0].kill()
        await p.reply("answer", to=voice)

    report = await run_scenario(
        app, _scenario(tmp_path), models="real", clock="real", real_audio=True, drain_s=0.5, ready_timeout_s=5
    )
    assert not report.passed
    assert not report.performance["timing_comparable"]
    if failure == "handler":
        assert report.performance["error_events"] > 0
    if failure == "missing_transcript":
        assert report.performance["handler_coverage"] == {"received": 0, "expected": 2}
        assert report.performance["completed_reply_text_s"]["n"] == 0
    if failure == "source":
        assert report.performance["execution_status"] == "failed"
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=owned[0][1], create=False)
    assert app._ready_file is None
    assert not app._participants


@pytest.mark.integration
async def test_teardown_errors_are_excluded_from_measurement(tmp_path, monkeypatch):
    from loguru import logger
    from xr_ai_app import Component

    app, voice = _app(tmp_path, monkeypatch)

    class Teardown(Component):
        async def on_stop(self):
            logger.error("teardown error outside measurement")

    app.add(Teardown())

    @app.on_turn
    async def answer(turn, p):
        await p.reply("answer", to=voice)

    result = await run_scenario(
        app, _scenario(tmp_path), models="real", clock="real", real_audio=True, drain_s=0.5, ready_timeout_s=5
    )
    assert all(step.ok for step in result.steps)
    assert result.performance["error_events"] == 0


def test_measurements_follow_turn_ids_and_exclude_startup_and_teardown():
    from types import SimpleNamespace

    from xr_ai_app.devkit._audio_report import _performance

    def event(kind, turn=None, **data):
        return SimpleNamespace(kind=kind, turn=turn, data=data)

    plan = {
        "events": [{"action": "audio", "index": 1, "handler": True}, {"action": "audio", "index": 2, "handler": True}]
    }
    source = [
        {
            "kind": "input",
            "action": "audio",
            "index": 1,
            "monotonic": 10,
            "audio_start_us": 1000000,
            "audio_end_us": 1100000,
            "handler": True,
            "text": "next",
        },
        {"kind": "speech_end", "input_index": 1, "monotonic": 10.1},
        {
            "kind": "input",
            "action": "audio",
            "index": 2,
            "monotonic": 11,
            "audio_start_us": 2000000,
            "audio_end_us": 2100000,
            "handler": True,
            "text": "next",
        },
        {"kind": "speech_end", "input_index": 2, "monotonic": 11.1},
        {"kind": "measurement_finished", "monotonic": 14, "max_sender_lateness_s": 0.001},
    ]
    observed = [
        (10.2, event("turn", "first", phase="start", timestamp_us=1000000)),
        (11.2, event("turn", "second", phase="start", timestamp_us=2000000)),
        (11.4, event("response", request_id="second", response_id="r2", final=True)),
        (11.5, event("turn", "second", phase="end", outcome="done")),
        (13.3, event("response", request_id="first", response_id="r1", final=True)),
        (13.4, event("turn", "first", phase="end", outcome="done")),
    ]
    diagnostics = [{"kind": "error", "monotonic": 9}, {"kind": "error", "monotonic": 15}]
    result = _performance(plan, source, observed, diagnostics, 10, 16, True)
    assert result["timing_comparable"]
    assert result["error_events"] == 0
    assert result["completed_reply_text_s"]["worst"] == pytest.approx(3.2)
    observed.append((13.4, event("response", request_id="first", response_id="unfinished", final=False)))
    result = _performance(plan, source, observed, diagnostics, 10, 16, True)
    assert not result["timing_comparable"]
    assert result["reply_coverage"] == {"completed": 1, "expected": 2}


@pytest.mark.integration
async def test_missing_media_does_not_start_app_or_create_output(tmp_path, monkeypatch):
    app, _voice = _app(tmp_path, monkeypatch)
    scenario = Scenario.from_dict({"steps": [{"join": "p1"}, {"audio": {"path": "absent.wav"}}]}, base_dir=tmp_path)
    output = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        await run_scenario(app, scenario, models="real", clock="real", real_audio=True, out=output)
    assert not output.exists()
    assert app._start_task is None


def test_cli_rejects_accelerated_audio_before_importing_app():
    from xr_ai_app.devkit.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["run", "absent.py", "scenario.yaml", "--real-audio", "--real-models", "--clock", "virtual"])
    assert exc.value.code == 2


@pytest.mark.integration
@pytest.mark.parametrize("failed_call", [False, True])
async def test_hosted_profile_uses_normal_adapter_and_keeps_credentials_out_of_report(
    tmp_path, monkeypatch, failed_call
):
    import httpx
    from xr_ai_models import _factory

    app, voice = _app(tmp_path, monkeypatch)
    profile_path = tmp_path / "models.json"
    raw = json.loads(profile_path.read_text())
    raw["models"]["frontier"] = {
        "category": "llm",
        "adapter": {"kind": "openai_compat", "model_name": "my-frontier-model"},
        "endpoint": {"base_url": "https://frontier.example", "api_key_env": "TEST_FRONTIER_KEY", "readiness": "none"},
    }
    raw["roles"] = {"answer": "frontier"}
    profile_path.write_text(json.dumps(raw))
    monkeypatch.setenv("TEST_FRONTIER_KEY", "private-test-secret")
    requests = []

    def respond(request):
        requests.append(request)
        if failed_call:
            return httpx.Response(401, json={"error": "injected endpoint failure"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "answer"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        original = _factory.OpenAICompatLLM
        monkeypatch.setattr(_factory, "OpenAICompatLLM", lambda **kwargs: original(client=client, **kwargs))
        role = app.role("answer")

        @app.on_turn
        async def answer(turn, p):
            try:
                text = await role.ask(None, turn.text)
            except httpx.HTTPStatusError:
                text = "answer fallback"
            await p.reply(text, to=voice)

        output = tmp_path / "run"
        result = await run_scenario(
            app,
            _scenario(tmp_path),
            models="real",
            clock="real",
            real_audio=True,
            drain_s=0.5,
            ready_timeout_s=5,
            out=output,
        )
    assert result.passed is (not failed_call), result.transcript()
    assert all(step.ok for step in result.steps)
    assert result.performance["timing_comparable"] is (not failed_call)
    assert (result.performance["error_events"] > 0) is failed_call
    assert len(requests) == 2
    assert all(request.url.host == "frontier.example" for request in requests)
    assert all(request.headers["authorization"] == "Bearer private-test-secret" for request in requests)
    assert all(json.loads(request.content)["model"] == "my-frontier-model" for request in requests)
    assert "private-test-secret" not in (output / "report.json").read_text()
    assert result.performance["configuration"]["roles"] == {"answer": "frontier"}


@pytest.mark.integration
async def test_cancellation_stops_only_the_owned_source(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from xr_ai_app.devkit import _audio

    app, _voice = _app(tmp_path, monkeypatch)
    context = _audio.multiprocessing.get_context("spawn")
    owned = []

    def process(**kwargs):
        result = context.Process(**kwargs)
        owned.append(result)
        return result

    monkeypatch.setattr(
        _audio.multiprocessing, "get_context", lambda _kind: SimpleNamespace(Pipe=context.Pipe, Process=process)
    )
    task = asyncio.create_task(
        run_scenario(
            app,
            _scenario(tmp_path),
            models="real",
            clock="real",
            real_audio=True,
            drain_s=10,
            ready_timeout_s=5,
            out=tmp_path / "interrupted",
        )
    )
    try:
        async with asyncio.timeout(5):
            while app._ready_file is None or not app._ready_file.exists():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert owned[0]._closed
        assert app._ready_file is None
        assert not app._participants
        saved = json.loads((tmp_path / "interrupted/report.json").read_text())
        assert not saved["passed"]
        assert saved["performance"]["execution_status"] == "failed"
        assert not saved["performance"]["timing_comparable"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_five_minute_sample_uses_the_same_scenario_contract(tmp_path):
    path = Path(__file__).resolve().parents[3] / "app-samples/conversation/eval/speech/realtime.yaml"
    scenario = Scenario.load(path)
    for step in scenario.steps:
        if step.action == "audio":
            _wav(tmp_path / step.value["path"])
    plan = _prepare_audio(scenario, tmp_path, 10)
    assert plan["duration_s"] == 300
    assert len([event for event in plan["events"] if event["action"] == "audio" and event["handler"]]) == 2


@pytest.mark.integration
async def test_effective_component_voice_gate_is_honored_and_snapshotted(tmp_path, monkeypatch):
    app, voice = _app(tmp_path, monkeypatch)
    custom = tmp_path / "custom-gate.yaml"
    content = (tmp_path / "voice_gate.yaml").read_bytes() + b"\n# component-specific gate\n"
    custom.write_bytes(content)
    voice._speech["gate"] = "custom-gate.yaml"

    @app.on_start
    async def change_original():
        custom.write_text("invalid: [")

    @app.on_turn
    async def answer(turn, p):
        await p.reply("answer", to=voice)

    result = await run_scenario(
        app, _scenario(tmp_path), models="real", clock="real", real_audio=True, drain_s=0.5, ready_timeout_s=5
    )
    assert result.passed, result.transcript()
    assert result.performance["configuration"]["voice_gate_sha256"] == hashlib.sha256(content).hexdigest()
    assert voice._speech["gate"] == "custom-gate.yaml"


@pytest.mark.integration
async def test_partial_audio_setup_restores_native_config(tmp_path, monkeypatch):
    from xr_ai_app.devkit import _audio

    app, voice = _app(tmp_path, monkeypatch)
    config, speech = app._config, voice._speech

    def unavailable(_kind):
        raise RuntimeError("injected process setup failure")

    monkeypatch.setattr(_audio.multiprocessing, "get_context", unavailable)
    with pytest.raises(RuntimeError, match="injected process setup failure"):
        await run_scenario(app, _scenario(tmp_path), models="real", clock="real", real_audio=True)
    assert app._config is config
    assert voice._speech is speech
    assert app._ready_file is None
    assert app._start_task is None


def test_cli_prints_coverage_and_errors_for_audio_runs(tmp_path, monkeypatch, capsys):
    from xr_ai_app.devkit import __main__ as cli
    from xr_ai_app.devkit.runner import Report

    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("name: terminal metrics\nsteps: []\n")
    metrics = {
        "execution_status": "completed",
        "timing_comparable": False,
        "handler_coverage": {"received": 1, "expected": 1},
        "reply_coverage": {"completed": 1, "expected": 1},
        "error_events": 1,
        "stt": {"drops": 0, "resets": 0},
        "max_sender_lateness_s": 0.002,
    }

    async def run(*args, **kwargs):
        assert kwargs["real_audio"]
        return Report("terminal metrics", True, performance=metrics)

    monkeypatch.setattr(cli, "load_app", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "run_scenario", run)
    assert cli.main(["run", "app.py", str(scenario), "--real-audio", "--real-models", "--clock", "real"]) == 0
    output = capsys.readouterr().out
    summary = json.loads(output.splitlines()[-1])
    assert summary["handler_coverage"] == metrics["handler_coverage"]
    assert summary["error_events"] == 1 and summary["timing_comparable"] is False


@pytest.mark.integration
@pytest.mark.parametrize("cancel", [False, True])
async def test_startup_interruption_drains_the_app_and_its_start_hook(tmp_path, monkeypatch, cancel):
    app, _voice = _app(tmp_path, monkeypatch)
    entered, finished, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @app.on_start
    async def hold():
        entered.set()
        try:
            await release.wait()
        finally:
            finished.set()

    task = asyncio.create_task(
        run_scenario(
            app,
            _scenario(tmp_path),
            models="real",
            clock="real",
            real_audio=True,
            ready_timeout_s=1,
            out=tmp_path / "startup",
        )
    )
    await asyncio.wait_for(entered.wait(), 5)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    else:
        report = await asyncio.wait_for(task, 5)
        assert not report.passed
        assert report.performance["execution_status"] == "failed"
    assert finished.is_set()
    assert app._start_task.done()
    assert not app._admitting and not app._app_tasks
    release.set()
    await asyncio.sleep(0)
    assert not app._admitting


@pytest.mark.integration
async def test_superseded_partial_stream_does_not_count_as_a_completed_reply(tmp_path, monkeypatch):
    app, voice = _app(tmp_path, monkeypatch)
    calls = 0

    async def partial():
        yield "answer fragment"
        await asyncio.Future()

    @app.on_turn
    async def answer(turn, p):
        nonlocal calls
        calls += 1
        await p.reply(partial() if calls == 1 else "answer complete", to=voice)

    report = await run_scenario(
        app,
        _scenario(tmp_path),
        models="real",
        clock="real",
        real_audio=True,
        ready_timeout_s=5,
        drain_s=1,
        out=tmp_path / "partial",
    )
    assert not report.passed
    assert report.performance["handler_coverage"] == {"received": 2, "expected": 2}
    assert report.performance["reply_coverage"] == {"completed": 1, "expected": 2}
    assert report.performance["inputs"][0]["completed_reply_text_s"] is None
    assert not report.performance["timing_comparable"]


@pytest.mark.parametrize("rate,channels", [(48000, 2), (44100, 1)])
def test_recorded_wav_conversion_preserves_duration_and_mixdown(rate, channels):
    from xr_ai_app.devkit._audio import _wav_samples

    body = io.BytesIO()
    samples = np.full((rate // 10, channels), 4000, dtype="<i2")
    if channels == 2:
        samples[:, 1] = -4000
    with wave.open(body, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    converted = _wav_samples(body.getvalue(), "recorded.wav")
    assert len(converted) == 1600
    assert converted.dtype == np.dtype("float32")
    if channels == 2:
        assert np.max(np.abs(converted)) == 0
    else:
        assert np.mean(converted[20:-20]) == pytest.approx(4000 / 32768, abs=0.001)


@pytest.mark.parametrize("issue", ["control", "unattributed", "drop", "reset"])
def test_comparability_rejects_control_turns_attribution_and_speech_disruption(issue):
    from types import SimpleNamespace

    from xr_ai_app.devkit._audio_report import _performance

    control = issue == "control"
    plan = {"events": [{"action": "audio", "index": 1, "handler": not control, "reply": False}]}
    source = [
        {
            "kind": "input",
            "action": "audio",
            "index": 1,
            "handler": not control,
            "reply": False,
            "audio_start_us": 1000000,
            "audio_end_us": 1100000,
            "monotonic": 10,
            "text": "hi",
        },
        {"kind": "speech_end", "input_index": 1, "monotonic": 10.1},
        {"kind": "measurement_finished", "monotonic": 12, "max_sender_lateness_s": 0},
    ]
    observed = [
        (
            10.2,
            SimpleNamespace(
                kind="turn",
                turn="t1",
                data={"phase": "start", "timestamp_us": 999999999 if issue == "unattributed" else 1000000},
            ),
        )
    ]
    speech = [{"kind": issue, "monotonic": 10.3}] if issue in {"drop", "reset"} else []
    result = _performance(plan, source, observed, speech, 10, 12, True)
    assert not result["timing_comparable"]
    assert result["unexpected_control_turns"] == int(control)
    assert result["unattributed_turns"] == int(issue == "unattributed")


@pytest.mark.integration
async def test_closing_leave_preserves_reply_during_drain(tmp_path, monkeypatch):
    import dataclasses

    from xr_ai_app.devkit.scenario import Step

    app, voice = _app(tmp_path, monkeypatch)
    original = _scenario(tmp_path)
    scenario = dataclasses.replace(original, steps=(*original.steps[:3], Step("leave", "p1", at=0.2)), duration_s=0.2)

    @app.on_turn
    async def answer(turn, p):
        await asyncio.sleep(0.3)
        await p.reply("answer during drain", to=voice)

    report = await run_scenario(
        app,
        scenario,
        models="real",
        clock="real",
        real_audio=True,
        drain_s=2,
        ready_timeout_s=5,
        out=tmp_path / "leave",
    )
    assert report.performance["reply_coverage"] == {"completed": 1, "expected": 1}
    rows = [json.loads(line) for line in (tmp_path / "leave/audio-events.jsonl").read_text().splitlines()]
    leave = next(row for row in rows if row["kind"] == "input" and row["action"] == "leave")
    assert leave["t"] >= 2.2
    assert rows.index(leave) < next(i for i, row in enumerate(rows) if row["kind"] == "measurement_finished")


@pytest.mark.integration
async def test_dead_voice_startup_fails_without_waiting_for_readiness_deadline(tmp_path, monkeypatch):
    app, _voice = _app(tmp_path, monkeypatch)

    class DeadVoice:
        async def run(self, _runtime):
            raise RuntimeError("bad speech endpoint")

    app._voice_factory = lambda _app: DeadVoice()
    report = await asyncio.wait_for(
        run_scenario(app, _scenario(tmp_path), models="real", clock="real", real_audio=True, ready_timeout_s=600), 8
    )
    assert not report.passed
    assert any("voice pipeline exited before readiness" in step.detail for step in report.steps)
    assert not app._participants and not app._app_tasks


@pytest.mark.integration
@pytest.mark.parametrize("update", [False, True])
async def test_unprompted_assertion_cannot_match_conversation_acknowledgement(tmp_path, monkeypatch, update):
    app, voice = _app(tmp_path, monkeypatch)
    _wav(tmp_path / "speech.wav")
    scenario = Scenario.from_dict({"duration_s": 0.5, "steps": [
        {"join": "p1"},
        {"at": 0.1, "audio": {"path": "speech.wav", "reply": not update}},
        {"at": 0.2, "expect": {"said": "red", "unprompted": True, "within_s": 1.5}},
    ]}, base_dir=tmp_path)

    @app.on_turn
    async def answer(turn, p):
        if update:
            await p.say("red", to=voice)
        else:
            await p.reply("red", to=voice)

    report = await run_scenario(app, scenario, models="real", clock="real", real_audio=True,
                                drain_s=2, ready_timeout_s=5)
    assert next(step for step in report.steps if step.action == "expect").ok is update


@pytest.mark.integration
@pytest.mark.parametrize("streaming", [False, True])
async def test_speech_replay_preserves_camera_mode(tmp_path, monkeypatch, streaming):
    import dataclasses

    app, voice = _app(tmp_path, monkeypatch)
    source = _scenario(tmp_path)
    scenario = dataclasses.replace(source, steps=tuple(
        dataclasses.replace(step, value={"path": step.value, "stream": streaming}) if step.action == "image" else step
        for step in source.steps
    ))
    pictures = []

    @app.on_turn
    async def answer(turn, p):
        if streaming:
            assert await p.frames._picture(1) is None
        frame = await p.frames.current()
        assert frame is not None
        pictures.append(frame)
        await p.reply("answer", to=voice)

    report = await run_scenario(app, scenario, models="real", clock="real", real_audio=True,
                                drain_s=2, ready_timeout_s=5, out=tmp_path / "on-demand")
    assert len(pictures) == 2
    assert report.performance["reply_coverage"] == {"completed": 2, "expected": 2}
    rows = [json.loads(line) for line in (tmp_path / "on-demand/audio-events.jsonl").read_text().splitlines()]
    assert any(row["kind"] == ("image_capture_rejected" if streaming else "image_capture") for row in rows)
    assert all(row["stream"] is streaming for row in rows if row["kind"] == "input" and row["action"] == "image")


@pytest.mark.integration
async def test_zero_time_welcome_is_observed_after_scheduled_join(tmp_path, monkeypatch):
    from xr_ai_app import Component

    app, voice = _app(tmp_path, monkeypatch)
    _wav(tmp_path / "speech.wav")

    class Welcome(Component):
        async def on_participant_enrolled(self, p):
            await p.say("welcome immediately", to=voice)

    app.add(Welcome())
    scenario = Scenario.from_dict({"duration_s": 0.6, "steps": [
        {"join": "p1"}, {"expect": {"said": "welcome immediately", "within_s": 0.5}},
        {"at": 0.3, "audio": {"path": "speech.wav"}},
    ]}, base_dir=tmp_path)

    @app.on_turn
    async def answer(turn, p):
        await p.reply("answer", to=voice)

    result = await run_scenario(app, scenario, models="real", clock="real", real_audio=True,
                                drain_s=1, ready_timeout_s=5)
    assert next(step for step in result.steps if step.action == "expect").ok
    welcome = next(output for output in result.outputs if output["text"] == "welcome immediately")
    assert welcome["t"] >= 0
