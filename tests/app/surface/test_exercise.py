# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise one piece of a sample on a window of a video, through its real code, the app unchanged.

The clip is synthetic (cv2-written, a few seconds of solid colour); the roles are scripted, so what is checked is the
harness: a unit runs sample 10's own camera step (an ``xr_ai_kit.sop`` step run on its own), frames from the window
reach it, and its outcome, the clip time it ended at and its model calls come back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xr_ai_app.devkit import Probe, exercise, resolve_target
from xr_ai_app.devkit.__main__ import main as devkit_main
from xr_ai_app.testing import FakeModels
from xr_ai_kit.sop import Run

BED = "A spool lies on the print bed."
CLEAR = "The print bed is in view and empty."
DOOR_OPEN = "The printer door is open."

UNIT_FILE = """
from dataclasses import dataclass

from xr_ai_app import trace


@dataclass
class Done:
    frames: int


async def three_frames(p, probe):
    n = 0
    async for _ in p.frames.every(0.5):
        n += 1
        await probe.report(f"frame {n}")
        if n == 3:
            trace.emit("unit", "three_frames", frames=n)
            return Done(n)
"""


def step_unit(step):  # noqa: ANN001, ANN201
    """A unit that runs one step of a procedure on its own, with a run as its io (as the runner would)."""

    async def unit(p, probe):  # noqa: ANN001, ANN202
        return await step.run(Run(p, 1, name="exercise"))

    return unit


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Eight seconds at 4 fps of a grey frame."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np  # noqa: PLC0415

    path = tmp_path_factory.mktemp("clip") / "demo.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (64, 48))
    for _ in range(32):
        writer.write(np.full((48, 64, 3), 128, dtype=np.uint8))
    writer.release()
    return path


async def test_a_step_runs_on_a_window_until_done(load_sample, clip: Path) -> None:
    m = load_sample("10-guided-procedure")
    models = FakeModels({"perceive": [BED, CLEAR, CLEAR], "decide": ["negative", "positive", "positive"]})
    unit = step_unit(m.bed_clear)
    run = await exercise(m.app, unit, target="bed_clear", video=clip, from_s=2.0, to_s=7.0, fps=2, models=models)
    assert run.status == "returned" and run.outcome == "Verification", run.to_json()
    assert run.result.status == "completed" and run.result.positives == 2
    assert run.ended_at_s is not None and 2.0 <= run.ended_at_s <= 7.0
    assert [call["name"] for call in run.calls] == ["perceive", "decide"] * 3
    assert models.calls("perceive")[0].prompt == m.bed_clear.step.check.question  # the watch step, beside its escalator


async def test_a_step_that_never_finishes_is_unfinished(load_sample, clip: Path) -> None:
    m = load_sample("10-guided-procedure")
    models = FakeModels({"perceive": [DOOR_OPEN] * 10, "decide": ["negative"] * 10})
    unit = step_unit(m.door_closed)
    run = await exercise(m.app, unit, video=clip, from_s=0.0, to_s=2.0, fps=2, models=models, linger_s=0.5)
    assert run.status == "unfinished" and run.outcome == "unfinished"
    assert len(models.calls("decide")) >= 1  # it was looking


async def test_unknown_targets_say_what_exists(load_sample) -> None:
    m = load_sample("05-manual-dispatch")
    with pytest.raises(ValueError, match="unknown target kind"):
        resolve_target(m.app, "nope:x")
    with pytest.raises(ValueError, match="fn targets are"):
        resolve_target(m.app, "fn:no_module")


async def test_a_router_destination_runs_on_one_turn(load_sample) -> None:
    m = load_sample("05-manual-dispatch")
    unit = resolve_target(m.app, "destination:on", {"text": "lights on please"})
    run = await exercise(m.app, unit, target="destination:on", models=FakeModels({}))
    assert run.status == "returned"
    assert [s["text"] for s in run.said] == ["Lights on."]
    assert any(e["kind"] == "destination" and e["name"] == "on" for e in run.events)


async def test_a_helper_destination_in_a_router_is_a_target(load_sample) -> None:
    m = load_sample("04-notes-mode")
    unit = resolve_target(m.app, "destination:conversation", {"text": "hello"})
    run = await exercise(
        m.app, unit, target="destination:conversation", models=FakeModels({"answer_with_reasoning": ["Hi."]})
    )
    assert [s["text"] for s in run.said] == ["Hi."]


async def test_a_mode_called_directly_still_classifies_the_turn(load_sample) -> None:
    m = load_sample("04-notes-mode")
    # no Router admission on this path: the mode classifies the turn itself
    unit = resolve_target(m.app, "destination:notes", {"text": "start taking notes"})
    run = await exercise(m.app, unit, target="destination:notes", models=FakeModels({"decide": ["start"]}))
    assert [s["text"] for s in run.said] == ["Okay, I'm taking notes."]  # started; not stored as a note ("Noted.")
    assert [e["data"]["action"] for e in run.events if e["kind"] == "mode"] == ["start"]


async def test_any_function_with_context_injected(load_sample, clip: Path) -> None:
    """``setup`` injects context before the unit runs; a function target sees the clip's frames."""
    m = load_sample("02-vision-qa")

    async def setup(p, probe: Probe) -> None:  # noqa: ANN001
        p.state.question = "what color is it?"

    async def unit(p, probe: Probe) -> tuple[str, int]:  # noqa: ANN001
        frame = await p.frames.current(max_age_s=5, timeout_s=5)
        await probe.report(f"saw {frame.width}x{frame.height}")
        return p.state.question, frame.width

    run = await exercise(m.app, unit, video=clip, from_s=1.0, to_s=2.0, setup=setup, models=FakeModels({}))
    assert run.result == ("what color is it?", 64)
    assert run.reports[0][1] == "saw 64x48" and run.reports[0][0] is not None


def test_cli_checks_the_outcome_and_the_window(load_sample, clip: Path, tmp_path: Path, capsys) -> None:  # noqa: ANN001
    sample = Path(__file__).resolve().parents[3] / "app-samples" / "sdk-tour" / "10-guided-procedure"
    units = tmp_path / "units.py"
    units.write_text(UNIT_FILE)
    fakes = tmp_path / "fakes.yaml"
    fakes.write_text(json.dumps({}))
    load_sample("10-guided-procedure")  # the guard profile, so nothing reaches a real endpoint
    args = [
        "exercise",
        str(sample / "app.py"),
        f"fn:{units}:three_frames",
        "--video",
        str(clip),
        "--from",
        "3",
        "--to",
        "7",
        "--fakes",
        str(fakes),
        "--expect",
        "Done",
        "--out",
        str(tmp_path / "out"),
    ]
    assert devkit_main([*args, "--between", "3", "7"]) == 0
    saved = json.loads((tmp_path / "out" / "exercise.json").read_text())
    assert saved["outcome"] == "Done" and saved["result"] == {"type": "Done", "frames": 3}
    assert (tmp_path / "out" / "trace.jsonl").read_text().strip()
    assert devkit_main([*args, "--between", "0", "1"]) == 1  # ended outside the annotated window
    capsys.readouterr()


async def test_a_clip_that_cannot_play_is_an_error_not_an_outcome(load_sample, tmp_path: Path) -> None:
    m = load_sample("10-guided-procedure")
    unit = step_unit(m.bed_clear)
    with pytest.raises(FileNotFoundError):
        await exercise(m.app, unit, video=tmp_path / "missing.mp4", models=FakeModels({}), linger_s=0.2)
