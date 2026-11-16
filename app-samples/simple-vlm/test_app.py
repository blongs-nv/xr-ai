# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scenarios on scripted models, and what the device saw besides speech.

Nothing here starts a model or a device hub: the local device plays each
scenario with the replies in its `<name>.fakes.yaml`.

Run from this directory: uv --config-file ../../uv.toml run pytest
"""

import importlib
from pathlib import Path

import app as module
import pytest
from xr_ai_app.devkit import (
    LocalDevice,
    Scenario,
    fakes_from_file,
    run_scenario,
)

SCENARIOS = Path(__file__).resolve().parent / "scenarios"


def play(name: str):
    """Play scenarios/<name>.yaml on a fresh app; return the report."""
    app = importlib.reload(module).app
    fakes = fakes_from_file(SCENARIOS / f"{name}.fakes.yaml")
    scenario = Scenario.load(SCENARIOS / f"{name}.yaml")
    return run_scenario(app, scenario, models=fakes)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["typed", "spoken", "on_demand"])
async def test_scenario_passes(name: str) -> None:
    report = await play(name)
    assert report.passed, report.transcript()


@pytest.mark.asyncio
async def test_without_a_stream_the_app_asks_for_one_picture() -> None:
    report = await play("on_demand")
    kinds = [i["kind"] for i in report.inputs]
    assert "frame" not in kinds  # nothing was streamed
    assert kinds.count("capture") == 1  # the picture the answer is about


@pytest.mark.asyncio
async def test_the_device_sees_processing_then_idle() -> None:
    app = importlib.reload(module).app
    fakes = fakes_from_file(SCENARIOS / "typed.fakes.yaml")
    async with LocalDevice(app, models=fakes) as device:
        await device.join()
        await device.show_image(SCENARIOS / "media" / "sign.png")
        await device.say("What does the sign say?")
        assert await device.wait_for("exit") is not None
        await device.settle()
        assert device.hub.agent_status("p1") == ["processing", "idle"]
