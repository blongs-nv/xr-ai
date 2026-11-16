# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local device: scenarios, media playback, the report and the HTTP endpoint (xr_ai_next.devkit)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import socket
from pathlib import Path

import httpx
import pytest
from xr_ai_next.devkit import LocalDevice, Scenario, fakes_from_file, run_scenario, run_virtual, serve
from xr_ai_next.devkit.__main__ import main as devkit_main
from xr_ai_next.testing import FakeModels

SAMPLES = Path(__file__).resolve().parents[3] / "xr-ai-next" / "samples" / "sdk-tour"

SCENARIOS = SAMPLES / "02-vision-qa" / "scenarios"


def _image(path: Path, color: str) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path)


def _video(path: Path, seconds: int = 3, fps: int = 10) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(seconds * fps):
        writer.write(np.full((48, 64, 3), i * 3 % 255, dtype=np.uint8))
    writer.release()


@pytest.fixture
def mug_scenario(tmp_path: Path) -> Path:
    """The worked example exactly as committed (its media too), copied so the run's reports stay in tmp."""
    shutil.copytree(SCENARIOS, tmp_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns("runs"))
    return tmp_path


async def test_scenario_passes_and_writes_a_report(load_sample, mug_scenario: Path) -> None:
    m = load_sample("02-vision-qa")
    report = await run_scenario(
        m.app,
        Scenario.load(mug_scenario / "mug.yaml"),
        models=fakes_from_file(mug_scenario / "mug.fakes.yaml"),
        out=mug_scenario / "runs",
    )
    assert report.passed, report.transcript()
    assert [o["text"] for o in report.outputs] == ["It's a red mug.", "You're welcome!"]
    transcript = (mug_scenario / "runs" / "transcript.md").read_text()
    assert "**p1**: What color is my mug?" in transcript and "PASSED" in transcript
    saved = json.loads((mug_scenario / "runs" / "report.json").read_text())
    assert saved["passed"] and len(saved["steps"]) == 7
    kinds = {json.loads(line)["kind"] for line in (mug_scenario / "runs" / "trace.jsonl").read_text().splitlines()}
    assert {"turn", "tool", "role", "output"} <= kinds


async def test_a_failed_check_fails_the_run(load_sample, tmp_path: Path) -> None:
    m = load_sample("01-hello-voice")
    scenario = Scenario.from_dict(
        {"steps": [{"join": "p1"}, {"say": "hi"}, {"expect": {"said": "goodbye", "within_s": 0.2}}]}
    )
    from xr_ai_next.testing import FakeModels

    report = await run_scenario(m.app, scenario, models=FakeModels({"answer_lightweight": ["hello!"]}))
    assert not report.passed
    (failure,) = report.failures
    assert failure.action == "expect" and "goodbye" in failure.detail


async def test_frames_and_video_reach_the_app(load_sample, tmp_path: Path) -> None:
    m = load_sample("02-vision-qa")
    for i, color in enumerate(["red", "green", "blue"]):
        _image(tmp_path / "frames" / f"{i:03d}.png", color)
    _video(tmp_path / "clip.avi")
    async with LocalDevice(m.app, speed=20.0) as device:
        await device.join("p1")
        assert await device.play_frames(tmp_path / "frames", fps=5) == 3
        p = m.app._participants.get("p1")
        first = await p.frames._fresh()
        assert first is not None and first.width == 64
        shown = await device.play_video(tmp_path / "clip.avi", fps=2, from_s=0, to_s=2)
        assert shown == 5
        assert [i["kind"] for i in device.inputs].count("frame") == 8


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def test_http_endpoint(load_sample, mug_scenario: Path) -> None:
    m = load_sample("02-vision-qa")
    port = _free_port()
    async with LocalDevice(m.app, models=fakes_from_file(mug_scenario / "mug.fakes.yaml")) as device:
        ready = asyncio.Event()
        server = asyncio.get_running_loop().create_task(serve(device, port=port, ready=ready))
        await ready.wait()
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                assert (await client.get("/health")).json() == {"ok": True}
                await client.post("/join", json={"participant": "p1"})
                frame = (mug_scenario / "media" / "mug.jpg").read_bytes()
                assert (await client.post("/frame?participant=p1", content=frame)).json() == {"ok": True}
                await client.post("/say", json={"text": "What color is my mug?"})
                out = (await client.get("/output", params={"after": 0, "wait_s": 3})).json()
                assert [o["text"] for o in out["outputs"]] == ["It's a red mug."]
                tools = (await client.get("/trace", params={"kind": "tool"})).json()["events"]
                assert tools[0]["name"] == "look"
                assert (await client.post("/nope")).status_code == 404
        finally:
            server.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server


def test_cli_run(load_sample, mug_scenario: Path, capsys: pytest.CaptureFixture[str]) -> None:
    app_file = SAMPLES / "02-vision-qa" / "app.py"
    code = devkit_main(
        ["run", str(app_file), str(mug_scenario / "mug.yaml"), "--fakes", str(mug_scenario / "mug.fakes.yaml")]
    )
    assert code == 0
    assert "PASSED" in capsys.readouterr().out


async def test_relay_export_writes_atof_and_atif(load_sample, mug_scenario: Path) -> None:
    pytest.importorskip("nemo_relay")
    m = load_sample("02-vision-qa")
    out = mug_scenario / "runs"
    report = await run_scenario(
        m.app,
        Scenario.load(mug_scenario / "mug.yaml"),
        models=fakes_from_file(mug_scenario / "mug.fakes.yaml"),
        out=out,
        relay=True,
    )
    assert report.passed
    (atif_file,) = (out / "atif").glob("*.json")
    trajectory = json.loads(atif_file.read_text())
    text = json.dumps(trajectory)
    assert "answer_with_reasoning" in text and "look" in text and "What color is my mug?" in text
    atof_lines = [line for f in (out / "atof").glob("*") for line in f.read_text().splitlines() if line.strip()]
    assert atof_lines and all(json.loads(line) for line in atof_lines)
    # every model call and tool is a child of its turn (each turn writes in its own forked Relay context)
    scopes = [e for e in map(json.loads, atof_lines) if e.get("kind") == "scope"]
    names = {e["uuid"]: e["name"] for e in scopes}
    starts = [e for e in scopes if e.get("scope_category") == "start"]
    calls = [e for e in starts if e.get("category") in ("llm", "tool")]
    parents = [(e["name"], names.get(e["parent_uuid"])) for e in calls]
    assert calls and all(parent == "turn" for _, parent in parents), parents
    assert sum(1 for e in starts if e["name"] == "turn") == 2


# ------------------------------------------------------------------------- evaluating with real media


def test_a_folder_of_scenarios_with_their_own_fakes_and_a_media_dir(load_sample, tmp_path: Path, capsys) -> None:  # noqa: ANN001
    """``run app.py scenarios/``: every scenario, each with ``<name>.fakes.yaml``; ``--media-dir`` is searched first
    (a recording under the same name replaces the committed placeholder); ``said_any`` accepts alternatives."""
    load_sample("02-vision-qa")  # the guard profile
    folder = tmp_path / "scenarios"
    folder.mkdir()
    shutil.copy(SCENARIOS / "mug.yaml", folder / "mug.yaml")
    shutil.copy(SCENARIOS / "mug.fakes.yaml", folder / "mug.fakes.yaml")
    (folder / "colors.yaml").write_text(
        "name: any color word\nsteps:\n  - join: p1\n  - image: media/mug.jpg\n  - say: What color is my mug?\n"
        "  - expect: {said_any: [blue, red], within_s: 5}\n"
    )
    (folder / "colors.fakes.yaml").write_text(
        "answer_with_reasoning:\n  - tool_call: {name: look, args: {question: color}}\n  - It's red.\nperceive: [red]\n"
    )
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    _image(recordings / "mug.jpg", "red")  # only in the media dir: the scenarios' own media/ does not exist
    app_file = Path(__file__).resolve().parents[3] / "xr-ai-next" / "samples" / "sdk-tour" / "02-vision-qa" / "app.py"
    out = tmp_path / "o"
    code = devkit_main(["run", str(app_file), str(folder), "--media-dir", str(recordings), "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "PASS  any color word  (colors.yaml)" in printed and "PASS  what color is my mug  (mug.yaml)" in printed
    assert (out / "mug" / "report.json").exists() and (out / "colors" / "transcript.md").exists()


def test_background_media_plays_while_later_steps_run(load_sample, tmp_path: Path) -> None:
    m = load_sample("01-hello-voice")
    for i in range(6):
        _image(tmp_path / "frames" / f"{i:03d}.png", "blue")
    scenario = Scenario.from_dict(
        {
            "steps": [
                {"join": "p1"},
                {"frames": {"dir": "frames", "fps": 2, "background": True}},
                {"say": "hi"},
                {"expect": {"said": "Hello", "within_s": 5}},
                {"leave": "p1"},
            ]
        },
        base_dir=tmp_path,
    )
    report = run_virtual(run_scenario(m.app, scenario, models=FakeModels({"answer_lightweight": ["Hello!"]})))
    assert report.passed, report.transcript()
    said = next(o["t"] for o in report.outputs)
    frames = [i["t"] for i in report.inputs if i["kind"] == "frame"]
    assert said < frames[-1] or len(frames) < 6  # the turn did not wait for the clip; leaving stopped it
    assert report.steps[1].detail.startswith(("stopped", "6 frames"))


def test_fakes_can_answer_by_prompt(tmp_path: Path) -> None:
    fakes = tmp_path / "f.yaml"
    fakes.write_text("r:\n  by_prompt:\n    - [weather, Sunny.]\n    - [time, Noon.]\n  otherwise: I don't know.\n")

    async def ask() -> list[str]:
        role = fakes_from_file(fakes).role("r")
        return [await role.ask(None, q) for q in ("what time is it", "the weather?", "who are you")]

    assert run_virtual(ask()) == ["Noon.", "Sunny.", "I don't know."]
