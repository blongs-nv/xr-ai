# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ProcedureCapture (xr_ai_procedures) and sample 12: a demonstration becomes an evidence packet.

The packet must stay readable by agent-samples/procedure-capture-guide's guide-generation skills, which read
``packet.json`` (status, counts, hierarchy, narration_status), ``summary.md``, ``frames/index.jsonl``,
``captions.jsonl`` (frame_path, activity, phase, delta), ``transcript.jsonl`` and ``errors.jsonl``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

from xr_ai_app.devkit import Scenario, fakes_from_file, run_scenario
from xr_ai_app.testing import FakeModels, solid_frame

SAMPLE = Path(__file__).resolve().parents[3] / "app-samples" / "sdk-tour" / "12-procedure-capture"

CAPTION = {
    "caption": "A kettle on a counter.",
    "delta": "Initial observation",
    "activity": "Making tea",
    "phase": "Filling the kettle",
}


def _load(load_sample, tmp_path: Path):  # noqa: ANN001, ANN202
    m = load_sample("12-procedure-capture")
    m.app._config = dataclasses.replace(m.app._config, artifacts_dir=tmp_path)  # packets go to tmp, not the sample
    return m


def _packet_dir(tmp_path: Path, participant: str = "p1") -> Path:
    (session,) = (tmp_path / participant / "sessions").iterdir()
    return session


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _until(check, timeout_s: float = 5.0) -> None:  # noqa: ANN001
    for _ in range(int(timeout_s / 0.02)):
        if check():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


async def test_sample_scenario_writes_a_complete_packet(load_sample, tmp_path: Path) -> None:
    m = _load(load_sample, tmp_path / "artifacts")
    scenario = Scenario.load(SAMPLE / "scenarios" / "demo.yaml")
    report = await run_scenario(m.app, scenario, models=fakes_from_file(SAMPLE / "scenarios" / "demo.fakes.yaml"))
    assert report.passed, report.transcript()
    packet_dir = _packet_dir(tmp_path / "artifacts")
    packet = json.loads((packet_dir / "packet.json").read_text())
    assert packet["schema_version"] == 1
    assert packet["status"] == "complete" and packet["narration_status"] == "complete"
    assert packet["counts"] == {"frames": 3, "transcripts": 2, "captions": 1}
    assert packet["stop_command"]["text"] == "stop capture"
    assert packet["hierarchy"][0]["name"] == "Making tea"
    assert packet["hierarchy"][0]["phases"][0]["name"] == "Filling the kettle"
    frames = _jsonl(packet_dir / "frames" / "index.jsonl")
    assert [f["frame_id"] for f in frames] == [1, 2, 3]
    assert all((packet_dir / f["path"]).is_file() for f in frames)
    (caption,) = _jsonl(packet_dir / "captions.jsonl")
    assert {"frame_path", "activity", "phase", "delta", "caption"} <= caption.keys()
    assert [t["text"] for t in _jsonl(packet_dir / "transcript.jsonl")] == [
        "first I fill the kettle to the line",
        "then I put it on the base and switch it on",
    ]
    summary = (packet_dir / "summary.md").read_text()
    assert "Status: complete" in summary and "**Heard:** first I fill the kettle to the line" in summary


async def test_captions_compare_with_the_previous_frame(load_sample, tmp_path: Path) -> None:
    m = _load(load_sample, tmp_path)
    m.capture._caption_every_s = 0.05
    second = {
        **CAPTION,
        "caption": "The kettle is on its base.",
        "delta": "Earlier: ... Current: ... Change: ...",
        "phase": "Switching on",
    }
    models = FakeModels({"perceive": [CAPTION, second]})
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("start capture")
        await m.capture.settled(s.participant())
        await s.show_frame(solid_frame("red"))
        await _until(lambda: len(models.calls("perceive")) == 1)
        await s.show_frame(solid_frame("green"))
        await _until(lambda: len(models.calls("perceive")) == 2)
        await s.say("stop capture")
        await m.capture.settled(s.participant())
        first_call, second_call = models.calls("perceive")
        assert "exactly ONE image" in first_call.prompt
        assert "exactly TWO images" in second_call.prompt and '"phase": "Filling the kettle"' in second_call.prompt
    captions = _jsonl(_packet_dir(tmp_path) / "captions.jsonl")
    assert [c["phase"] for c in captions] == ["Filling the kettle", "Switching on"]
    assert captions[1]["delta_reference_frame_id"] == captions[0]["frame_id"]


async def test_a_failed_caption_is_recorded_and_the_capture_goes_on(load_sample, tmp_path: Path) -> None:
    m = _load(load_sample, tmp_path)
    models = FakeModels({})  # the vision role has no reply: every caption fails
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("start capture")
        await m.capture.settled(s.participant())
        await s.show_frame(solid_frame("red"))
        await _until(lambda: (tmp_path / "p1").exists() and any((tmp_path / "p1").rglob("errors.jsonl")))
        await s.say("I am doing the first step")
        await s.say("stop capture")
        await m.capture.settled(s.participant())
    packet_dir = _packet_dir(tmp_path)
    (error,) = _jsonl(packet_dir / "errors.jsonl")
    assert error["kind"] == "caption_error"
    packet = json.loads((packet_dir / "packet.json").read_text())
    assert packet["status"] == "complete"
    assert packet["counts"] == {"frames": 1, "transcripts": 1, "captions": 0}


async def test_leaving_seals_an_open_capture(load_sample, tmp_path: Path) -> None:
    m = _load(load_sample, tmp_path)
    async with m.app.test_session(models=FakeModels({"perceive": [CAPTION]})) as s:
        await s.join()
        await s.say("start capture")
        await s.say("narration before leaving")
        await m.capture.settled(s.participant())
        await s.leave()
    packet = json.loads((_packet_dir(tmp_path) / "packet.json").read_text())
    assert packet["status"] == "complete" and packet["counts"]["transcripts"] == 1
    assert packet["stop_command"] is None  # sealed by the leave, not by a spoken stop


async def test_commands_apply_in_order_and_repeat(load_sample, tmp_path: Path) -> None:
    """Start and stop accepted back to back still produce one sealed packet; a second capture is a new packet;
    speech while idle is ignored (no reply, no packet)."""
    m = _load(load_sample, tmp_path)
    async with m.app.test_session(models=FakeModels({})) as s:
        await s.join()
        await s.say("hello there")  # idle: ignored
        await s.say("start capture")
        await s.say("Stop capture.")
        await s.say("start capture")
        await s.say("stop capture")
        await m.capture.settled(s.participant())
        assert s.spoken()[1:] == ["Capture started.", "Capture stopped.", "Capture started.", "Capture stopped."]
        sealed = m.capture.sealed(s.participant())
        assert [c.status for c in sealed] == ["complete", "complete"]
    assert len(list((tmp_path / "p1" / "sessions").iterdir())) == 2


async def test_a_capture_without_camera_frames_says_so(load_sample, tmp_path: Path) -> None:
    m = _load(load_sample, tmp_path)
    m.capture._first_frame_s = 0.05
    async with m.app.test_session(models=FakeModels({})) as s:
        await s.join()
        await s.say("start capture")
        await m.capture.settled(s.participant())
        await _until(lambda: any("camera frames" in t for t in s.spoken()))  # no frame came, no picture either
        await s.say("stop capture")
        await m.capture.settled(s.participant())
    assert s.spoken()[-3:] == [
        "Capture started.",
        "I'm not getting any camera frames, so nothing is being recorded. Turn on the camera to record.",
        "Capture stopped.",
    ]


async def test_a_capture_with_live_video_does_not_warn(load_sample, tmp_path: Path) -> None:
    m = _load(load_sample, tmp_path)
    m.capture._first_frame_s = 0.2
    async with m.app.test_session(models=FakeModels({"perceive": [CAPTION] * 5})) as s:
        await s.join()
        await s.say("start capture")
        await m.capture.settled(s.participant())
        await s.show_frame(solid_frame("red"))
        await asyncio.sleep(0.4)
        await s.say("stop capture")
        await m.capture.settled(s.participant())
    assert not any("camera frames" in t for t in s.spoken())
