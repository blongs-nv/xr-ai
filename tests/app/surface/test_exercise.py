# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise one piece of a sample on a window of a video, through its real code, the app unchanged.

The clip is synthetic (cv2-written, a few seconds of solid colour); the vision role is scripted, so what is checked
is the harness: the step is built as the procedure would, frames from the window reach it, its outcome and the clip
time it ended at come back, and so do its report lines and model calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xr_ai_app.devkit import Probe, exercise, resolve_target
from xr_ai_app.devkit.__main__ import main as devkit_main
from xr_ai_app.testing import FakeModels

BED = {"seen": "The bed has a spool on it.", "state": "objects_on_it"}
CLEAR = {"seen": "The bed is clear.", "state": "clear"}
DOOR_OPEN = {"seen": "The door is open.", "state": "open"}


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
    models = FakeModels({"perceive": [BED, CLEAR, CLEAR, CLEAR, CLEAR]})
    unit = resolve_target(m.app, "step:bed_clear")
    run = await exercise(m.app, unit, target="step:bed_clear", video=clip, from_s=2.0, to_s=7.0, fps=2, models=models)
    assert run.status == "returned" and run.outcome == "Done", run.to_json()
    assert run.ended_at_s is not None and 2.0 <= run.ended_at_s <= 7.0
    assert [text for _, text in run.reports][:2] == ["The bed has a spool on it.", "The bed is clear."]
    assert len(run.calls) == len(models.calls("perceive")) >= 3
    assert all(call["name"] == "perceive" for call in run.calls)


async def test_a_step_that_never_finishes_is_unfinished(load_sample, clip: Path) -> None:
    m = load_sample("10-guided-procedure")
    models = FakeModels({"perceive": [DOOR_OPEN] * 10})
    unit = resolve_target(m.app, "step:door_closed")
    run = await exercise(m.app, unit, video=clip, from_s=0.0, to_s=2.0, fps=2, models=models, linger_s=0.5)
    assert run.status == "unfinished" and run.outcome == "unfinished"
    assert [text for _, text in run.reports][0] == "The door is open."  # it was looking


async def test_unknown_targets_say_what_exists(load_sample) -> None:
    m = load_sample("10-guided-procedure")
    with pytest.raises(ValueError, match="bed_clear"):
        resolve_target(m.app, "step:no_such_step")
    with pytest.raises(ValueError, match="no facts model"):
        resolve_target(m.app, "step:bed_clear", {"facts": {"warm_up": {"x": 1}}})
    with pytest.raises(ValueError, match="unknown target kind"):
        resolve_target(m.app, "nope:x")


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
    fakes = tmp_path / "fakes.yaml"
    fakes.write_text(json.dumps({"perceive": [CLEAR] * 6}))
    load_sample("10-guided-procedure")  # the guard profile, so nothing reaches a real endpoint
    args = [
        "exercise",
        str(sample / "app.py"),
        "step:bed_clear",
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
    assert saved["outcome"] == "Done" and saved["result"]["type"] == "Done"
    assert (tmp_path / "out" / "trace.jsonl").read_text().strip()
    assert devkit_main([*args, "--between", "0", "1"]) == 1  # ended outside the annotated window
    capsys.readouterr()


async def test_a_clip_that_cannot_play_is_an_error_not_an_outcome(load_sample, tmp_path: Path) -> None:
    m = load_sample("10-guided-procedure")
    unit = resolve_target(m.app, "step:bed_clear")
    with pytest.raises(FileNotFoundError):
        await exercise(m.app, unit, video=tmp_path / "missing.mp4", models=FakeModels({}), linger_s=0.2)
