# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A procedure seen from the rest of the app: its events, its controls in code, and a whole run on a recording.

Runs on the devkit's virtual clock (steps look once a second; a 90 s timer is a step of sample 10), with scripted
camera reads, on sample 10 unchanged.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from xr_ai_next.devkit import exercise, resolve_target, run_virtual
from xr_ai_next.devkit.__main__ import main as devkit_main
from xr_ai_next.testing import FakeModels, solid_frame

BED = {"seen": "A spool on the bed.", "state": "objects_on_it"}
CLEAR = {"seen": "The bed is clear.", "state": "clear"}
OPEN = {"seen": "The door is open.", "state": "open"}
CLOSED = {"seen": "The door is closed.", "state": "closed"}
READS = [BED, CLEAR, CLEAR, OPEN, CLOSED, CLOSED]


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cv2 = pytest.importorskip("cv2")
    import numpy as np  # noqa: PLC0415

    path = tmp_path_factory.mktemp("clip") / "demo.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (64, 48))
    for _ in range(48):  # 12 s
        writer.write(np.full((48, 64, 3), 128, dtype=np.uint8))
    writer.release()
    return path


def test_events_and_controls_in_code(load_sample) -> None:
    m = load_sample("10-guided-procedure")
    seen: list[tuple[str, str | None]] = []
    m.procedure.on("*", lambda event, p: seen.append((event.kind, event.step)))

    @m.procedure.on("step_done")
    def broken(event, p) -> None:  # noqa: ANN001
        raise RuntimeError("a listener failed")  # traced, never stops the run

    async def scenario() -> list[str]:
        models = FakeModels({"perceive": [BED, CLEAR, CLEAR], "answer_lightweight": ["ok"] * 5})
        async with m.app.test_session(models=models) as s:
            await s.join()
            p = s.participant()
            run = await m.procedure.start(p)  # no turn, no decision model
            for _ in range(40):
                await s.show_frame(solid_frame("red"))
                if run.outcomes.get("bed_clear"):
                    break
                await asyncio.sleep(1)
            assert m.procedure.next(p)  # skip the door step
            for _ in range(20):
                if run.index >= 2:
                    break
                await asyncio.sleep(0.5)
            assert await m.procedure.stop(p)
            assert not m.procedure.next(p)  # nothing running any more
            return [run.status, run.outcomes.get("bed_clear"), run.outcomes.get("door_closed")]

    assert run_virtual(scenario()) == ["stopped", "done", "skipped"]
    assert seen[0] == ("run_start", None)
    assert ("step_start", "bed_clear") in seen and ("step_done", "bed_clear") in seen
    assert ("step_done", "door_closed") in seen and seen[-1] == ("run_end", None)


def _run_procedure(load_sample, clip: Path, annotations: dict | None):  # noqa: ANN001, ANN202
    m = load_sample("10-guided-procedure")
    options = {"annotations": annotations} if annotations else {}
    unit = resolve_target(m.app, "procedure:", options)
    models = FakeModels({"perceive": READS, "answer_lightweight": ["ok"] * 5})
    return run_virtual(
        exercise(m.app, unit, target="procedure:", video=clip, from_s=0.0, to_s=8.0, fps=2, models=models)
    )


def test_a_whole_procedure_on_a_recording_against_its_windows(load_sample, clip: Path) -> None:
    run = _run_procedure(load_sample, clip, {"bed_clear": [1.0, 3.5], "door_closed": [3.0, 6.0]})
    assert run.status == "returned", run.to_json()
    steps = {s.step: s for s in run.result.steps}
    assert steps["bed_clear"].outcome == "done" and steps["bed_clear"].verdict == "ok", run.to_json()
    assert steps["door_closed"].outcome == "done" and steps["door_closed"].verdict == "ok", run.to_json()
    assert steps["warm_up"].window is None and steps["warm_up"].verdict is None  # not annotated
    assert run.result.passed and run.passed
    kinds = [e["kind"] for e in run.result.events]
    assert kinds[0] == "run_start" and kinds.count("step_done") >= 2


def test_steps_outside_their_windows_fail(load_sample, clip: Path) -> None:
    run = _run_procedure(load_sample, clip, {"bed_clear": [5.0, 7.0], "door_closed": [0.0, 1.0]})
    steps = {s.step: s for s in run.result.steps}
    assert steps["bed_clear"].verdict == "early" and steps["door_closed"].verdict == "late"
    assert not run.result.passed and not run.passed


def test_unknown_annotated_steps_are_refused(load_sample) -> None:
    m = load_sample("10-guided-procedure")
    with pytest.raises(ValueError, match="unknown steps"):
        resolve_target(m.app, "procedure:", {"annotations": {"no_such_step": [0, 1]}})


def test_cli_exit_code_follows_the_windows(load_sample, clip: Path, tmp_path: Path, capsys) -> None:  # noqa: ANN001
    sample = Path(__file__).resolve().parents[3] / "xr-ai-next" / "samples" / "sdk-tour" / "10-guided-procedure"
    load_sample("10-guided-procedure")  # the guard profile
    fakes = tmp_path / "fakes.yaml"
    fakes.write_text(json.dumps({"perceive": READS, "answer_lightweight": ["ok"] * 5}))
    good, bad = tmp_path / "good.yaml", tmp_path / "bad.yaml"
    good.write_text("bed_clear: [1.0, 3.5]\ndoor_closed: [3.0, 6.0]\n")
    bad.write_text("bed_clear: [6.0, 7.0]\n")
    args = [
        "exercise",
        str(sample / "app.py"),
        "procedure:",
        "--video",
        str(clip),
        "--from",
        "0",
        "--to",
        "8",
        "--fakes",
        str(fakes),
    ]
    assert devkit_main([*args, "--option", f"annotations=@{good}", "--out", str(tmp_path / "out")]) == 0
    saved = json.loads((tmp_path / "out" / "exercise.json").read_text())
    assert [s["verdict"] for s in saved["result"]["steps"]][:2] == ["ok", "ok"]
    assert devkit_main([*args, "--option", f"annotations=@{bad}"]) == 1
    capsys.readouterr()
