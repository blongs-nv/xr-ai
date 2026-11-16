# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scenarios on scripted models, and what they show besides speech.

Nothing here starts a model or a device hub: the local device plays each
scenario with the replies in its `<name>.fakes.yaml`.

Run from this directory: uv --config-file ../../uv.toml run pytest
"""

import dataclasses
import importlib
import json
from pathlib import Path

import app as module
import pytest
from tea import DisplayFacts, Thermometer, assessable
from xr_ai_app.devkit import (
    Scenario,
    fakes_from_file,
    run_scenario,
    run_virtual,
)

SCENARIOS = Path(__file__).resolve().parent / "scenarios"


def play(name: str, tmp_path: Path):
    """Play scenarios/<name>.yaml on a fresh app, on virtual time (the
    countdown takes moments); the report and the fakes."""
    loaded = importlib.reload(module)
    # The run's record goes to tmp_path, not into the sample's folder.
    app = loaded.app
    config = app._config  # noqa: SLF001
    replaced = dataclasses.replace(config, artifacts_dir=tmp_path)
    app._config = replaced  # noqa: SLF001
    fakes = fakes_from_file(SCENARIOS / f"{name}.fakes.yaml")
    scenario = Scenario.load(SCENARIOS / f"{name}.yaml")
    report = run_virtual(run_scenario(loaded.app, scenario, models=fakes))
    return report, fakes


def test_the_guide_runs_start_to_finish(tmp_path: Path) -> None:
    report, _ = play("guide", tmp_path)
    assert report.passed, report.transcript()


def test_the_conversation_answers_from_the_guides_state(tmp_path: Path) -> None:
    report, fakes = play("guide", tmp_path)
    assert report.passed, report.transcript()
    asked = [
        c
        for c in fakes.calls("answer_with_reasoning")
        if "What step am I on" in (c.prompt or "") + (c.system or "")
    ]
    assert asked, "the question did not reach the conversation"
    assert "Tea guidance (run 1): running" in asked[0].system


def test_each_change_of_the_run_is_recorded(tmp_path: Path) -> None:
    report, _ = play("guide", tmp_path)
    assert report.passed, report.transcript()
    (record,) = tmp_path.glob("*/sop-tea.jsonl")
    ops = [json.loads(line) for line in record.read_text().splitlines()]
    steps = [(r["step"], r["status"]) for r in ops if r["op"] == "step"]
    assert steps[-2:] == [("steep", "running"), ("steep", "done")]
    assert [r["op"] for r in ops][-1] == "completed"


@pytest.mark.parametrize(
    ("reading", "unit", "ok"),
    [
        ({"value": 176, "unit_text": "°F"}, None, True),
        ({"value": 176}, None, False),  # no unit printed or known
        ({"value": 176}, "fahrenheit", True),  # the kettle's known unit
        ({"value": 80, "unit_text": "°C"}, "fahrenheit", True),  # known wins
        ({"value": 90, "unit_text": "°C", "role": "setpoint"}, None, False),
    ],
)
def test_only_a_measured_reading_with_a_unit_is_judged(
    reading, unit, ok
) -> None:
    known = DisplayFacts(unit=unit) if unit else None
    assert assessable(Thermometer(**reading), known) is ok
