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
from steps.heat_water.schemas import (
    DisplayCaption,
    DisplayFacts,
    HeatWaterInputs,
    reached_target,
)
from tea import WaterTemperature
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


def _caption(digits: str | None, letter: str | None = None) -> DisplayCaption:
    unit = (
        {"unit_letter_visible": True, "letter": letter}
        if letter
        else {"unit_letter_visible": False}
    )
    return DisplayCaption.model_validate(
        {
            "look": {
                "display_in_view": True,
                "display_is_on": "kettle",
                "display_kind": "lit digital digits",
                "where_in_image": "center",
                "text_on_display": [f"{digits}."],
                "reading": {
                    "evidence": "the display",
                    "shown_as": f"{digits}.",
                    "digits": digits,
                },
                "unit": unit,
                "marked_as": {"evidence": "nothing", "answer": "not shown"},
            }
        }
    )


@pytest.mark.parametrize(
    ("caption", "unit", "label"),
    [
        (_caption("176", "F"), None, "positive"),
        (_caption("176"), None, "unknown"),  # no unit printed or known
        (_caption("176"), "fahrenheit", "positive"),  # the unit the person gave
        (_caption("150"), "fahrenheit", "negative"),
        (
            _caption("80", "C"),
            "fahrenheit",
            "negative",
        ),  # the person's unit wins
        (_caption(None), "fahrenheit", "unknown"),  # digits not legible
    ],
)
def test_the_display_is_judged_in_code_only_with_a_number_and_its_unit(
    caption, unit, label
) -> None:
    known = DisplayFacts(unit=unit) if unit else None
    inputs = HeatWaterInputs(target=WaterTemperature(value=80, unit="celsius"))
    assert reached_target(caption, known, inputs) == label
