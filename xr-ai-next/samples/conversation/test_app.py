# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scenarios on scripted models, and what they show besides speech.

Nothing here starts a model or a device hub: the local device plays each
scenario with the replies in its `<name>.fakes.yaml`.

Run from this directory: uv --config-file ../../../uv.toml run pytest
"""

import importlib
from pathlib import Path

import app as module
import pytest
from xr_ai_next.devkit import (
    LocalDevice,
    Scenario,
    fakes_from_file,
    run_scenario,
)
from xr_ai_next.testing import FakeModels

SCENARIOS = Path(__file__).resolve().parent / "scenarios"
NAMES = ["general", "view", "view_on_demand", "notes", "alert"]


async def play(name: str):
    """Play scenarios/<name>.yaml on a fresh app; the report and the fakes."""
    app = importlib.reload(module).app
    fakes = fakes_from_file(SCENARIOS / f"{name}.fakes.yaml")
    scenario = Scenario.load(SCENARIOS / f"{name}.yaml")
    return await run_scenario(app, scenario, models=fakes), fakes


@pytest.mark.asyncio
@pytest.mark.parametrize("name", NAMES)
async def test_scenario_passes(name: str) -> None:
    report, _ = await play(name)
    assert report.passed, report.transcript()


@pytest.mark.asyncio
async def test_without_a_stream_the_view_takes_one_picture() -> None:
    report, _ = await play("view_on_demand")
    kinds = [i["kind"] for i in report.inputs]
    assert "frame" not in kinds  # nothing was streamed
    assert kinds.count("capture") == 1  # the picture the answer is about


@pytest.mark.asyncio
async def test_the_conversation_reads_the_note_from_its_context() -> None:
    report, fakes = await play("notes")
    assert report.passed, report.transcript()
    answers = fakes.calls("llm")
    assert len(answers) == 2  # both questions about the meeting
    for call in answers:
        assert "## notes" in call.system  # the system prompt's context
        assert "moved to Thursday at 3 pm" in call.system


@pytest.mark.asyncio
async def test_the_alert_is_urgent_spoken_once_and_checks_each_frame() -> None:
    report, fakes = await play("alert")
    assert report.passed, report.transcript()
    alerts = [o for o in report.outputs if o["text"] == "Alert"]
    assert len(alerts) == 1 and alerts[0]["urgent"]
    checks = [c for c in fakes.calls("router") if c.frames]
    assert len(checks) == 2  # the CLOSED frame, then the OPEN one
    assert "the sign says OPEN" in checks[0].context


@pytest.mark.asyncio
async def test_a_cancelled_alert_stops_watching() -> None:
    app = importlib.reload(module).app
    fakes = FakeModels(
        {
            "router": ["alerts", "start", "alerts", "cancel"],
            "llm": ['{"status": "ready", "condition": "the sign says OPEN"}'],
        }
    )
    async with LocalDevice(app, models=fakes) as device:
        await device.join()
        p = app.participants[0]
        await device.say("Let me know when the sign says OPEN.")
        assert await device.wait_for("alert you") is not None
        await device.settle()
        assert module.alert.active(p)
        await device.say("Cancel the alert.")
        assert await device.wait_for("cancelled") is not None
        assert not module.alert.active(p)
        await device.show_image(SCENARIOS / "media" / "sign_open.png")
        await device.settle()
    assert not [c for c in fakes.calls("router") if c.frames]  # no check
