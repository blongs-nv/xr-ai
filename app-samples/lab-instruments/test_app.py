# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scenarios on scripted models, and what they show besides speech.

Nothing here starts a model or a device hub: the local device plays each
scenario with the replies in its `<name>.fakes.yaml`. Marker detection runs
for real on the scenario pictures. The scenarios run on the devkit's virtual
clock, so the monitors' intervals (5 s, 30 s to call an instrument lost) take
no time.

Run from this directory: uv --config-file ../../uv.toml run pytest
"""

import asyncio
import importlib
import io
import json
from pathlib import Path

import app as module
import pytest
from PIL import Image
from xr_ai_app.devkit import (
    Scenario,
    fakes_from_file,
    run_scenario,
    run_virtual,
)

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "scenarios"
NAMES = [
    "read_one",
    "read_two",
    "view",
    "unknown_marker",
    "no_marker",
    "unreadable",
    "monitor_change",
    "monitor_lost",
    "watch",
]


def play(name: str):
    """Play scenarios/<name>.yaml on a fresh app: (report, fakes, app)."""
    app = importlib.reload(module).app
    fakes = fakes_from_file(SCENARIOS / f"{name}.fakes.yaml")
    scenario = Scenario.load(SCENARIOS / f"{name}.yaml")
    report = run_virtual(run_scenario(app, scenario, models=fakes))
    return report, fakes


def said(report, text: str) -> list[str]:
    return [o["text"] for o in report.outputs if text in o["text"]]


def records(name: str) -> list[dict]:
    """The newest session's `<name>.jsonl` of participant p1."""
    sessions = (HERE / "artifacts" / "p1").iterdir()
    newest = max(sessions, key=lambda d: d.name)
    lines = (newest / f"{name}.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


@pytest.mark.parametrize("name", NAMES)
def test_scenario_passes(name: str) -> None:
    report, _ = play(name)
    assert report.passed, report.transcript()


def test_one_model_call_reads_the_marked_frame_and_is_the_answer() -> None:
    report, fakes = play("read_one")
    (ask,) = fakes.calls("vlm")
    assert ask.frame.mime_type == "image/png"  # the marked copy
    pixels = Image.open(io.BytesIO(ask.frame.image)).convert("RGB")
    assert (255, 0, 255) in {pixels.getpixel((x, 480)) for x in range(1280)}
    assert '["magenta"]' in ask.prompt
    assert "colored X marker" in ask.system  # upstream's instrument prompt
    # The tool's reply ended the tool loop: the LLM was asked once.
    assert len(fakes.calls("llm")) == 1
    assert said(report, "Device1: 23.5 C")


def test_without_a_stream_the_reader_takes_one_picture() -> None:
    report, _ = play("read_two")
    kinds = [i["kind"] for i in report.inputs]
    assert "frame" not in kinds and kinds.count("capture") == 1


@pytest.mark.parametrize("name", ["unknown_marker", "no_marker"])
def test_no_configured_marker_means_no_model_call(name: str) -> None:
    report, fakes = play(name)
    assert report.passed and fakes.calls("vlm") == []


def test_the_monitor_speaks_once_per_change_and_records_it() -> None:
    report, fakes = play("monitor_change")
    assert len(said(report, "Now tracking Device1 at 23.5 C")) == 1
    assert len(said(report, "changed from 23.5 C to 24.1 C")) == 1
    # What the monitor said is in the conversation's context afterwards.
    stop = fakes.calls("llm")[1]
    assert "## monitoring history" in stop.system
    assert "Now tracking Device1 at 23.5 C." in stop.system
    events = records("instrument-monitoring")
    assert events[0]["type"] == "session"
    changes = [e for e in events if e.get("event_type") == "change"]
    assert [c["change_type"] for c in changes] == [
        "discovered",
        "reading_changed",
    ]
    assert changes[1]["previous_reading"] == "23.5 C"
    assert changes[1]["marker_type"] == "qr_code"
    assert events[-1]["type"] == "session_end"
    turns = [r for r in records("foreground") if "query" in r]
    assert [r["tools"] for r in turns] == [
        ["lab_instruments__start"],
        ["lab_instruments__stop"],
    ]
    assert turns[0]["response"] == "Lab instrument monitoring started."


def test_a_lost_instrument_is_said_once() -> None:
    report, _ = play("monitor_lost")
    assert len(said(report, "no longer tracking Device1")) == 1
    lost = [
        e
        for e in records("instrument-monitoring")
        if e.get("event_type") == "lost"
    ]
    assert len(lost) == 1 and lost[0]["meter_reading"] == "23.5 C"


def test_the_visual_monitor_is_silent_and_the_model_reads_its_looks() -> None:
    report, fakes = play("watch")
    assert not said(report, "thermometer")  # the looks were not spoken
    history = fakes.calls("llm")[2].prompt  # after recent_visual_history
    assert "The display now shows 24.1 C." in history
    looks = [r for r in records("monitor") if "record_type" in r]
    assert looks[0]["record_type"] == "baseline"
    assert looks[1]["changed"] is True


def test_a_view_answer_cut_off_stops_the_vision_stream() -> None:
    closed: list[bool] = []

    async def cosmos():
        try:
            yield "The image shows "
            yield "two displays."
        finally:
            closed.append(True)

    async def cut_off() -> None:
        answer = module._or_no_description(cosmos())
        assert await anext(answer) == "The image shows "
        await answer.aclose()
        assert closed == [True]  # now, not when garbage is collected

    asyncio.run(cut_off())
