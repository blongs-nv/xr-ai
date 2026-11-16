# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""step_eval.py on generated frames and scripted replies: the plumbing."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_eval  # noqa: E402

LABELS = """
recording: synthetic
margin_s: 0.5
display_unit: fahrenheit
target_c: 80
steps:
  fill_water: {no_data: true}
  heat_water:
    watch: [0, 20]
    negative: [[0, 10]]
    positive: [[10, 20]]
    traps: [[4, 6]]
"""


@pytest.fixture
def frames(tmp_path: Path) -> Path:
    rows = ["source_time_s,frame_image"]
    for t in range(20):
        name = f"f{t:03d}.jpg"
        Image.new("RGB", (32, 24), "gray").save(tmp_path / name)
        rows.append(f"{t},{name}")
    (tmp_path / "index.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_scores_frames_by_label_and_replays_the_live_rule(
    frames: Path, tmp_path: Path
) -> None:
    labels = tmp_path / "labels.yaml"
    labels.write_text(LABELS)
    fakes = tmp_path / "fakes.yaml"
    # Every reading is 176 with no unit; decide: always positive.
    fakes.write_text(
        "perceive:\n  by_prompt:\n    - [Transcribe the number, {value: 176}]\n"
        "decide:\n  by_prompt:\n    - [Has the water reached, positive]\n"
    )
    out = tmp_path / "out"
    code = step_eval.main(
        [
            "--labels",
            str(labels),
            "--frames-index",
            str(frames / "index.csv"),
            "--fakes",
            str(fakes),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["fill_water"] == {"no_data": True, "note": ""}
    heat = summary["heat_water"]
    # The kettle's unit (fahrenheit) makes a unitless reading assessable,
    # so every frame is judged positive: negatives become false positives,
    # the trap is judged positive, and the live rule completes early.
    assert heat["score"]["recall"] == 1.0
    assert heat["score"]["false_positive_rate"] == 1.0
    assert heat["score"]["traps_judged_positive"] == 1
    assert heat["replay"]["verdict"] == "early"
    lines = (out / "frames.jsonl").read_text().splitlines()
    assert len(lines) == 10  # every 2 s in [0, 20)
    assert "| heat_water |" in (out / "summary.md").read_text()
