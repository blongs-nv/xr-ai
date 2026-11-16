# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The packet harness: its scoring rules, and one plumbing run."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

WORDS = {
    "identify": ["label"],
    "fill_water": ["fill", "water"],
    "done": ["done"],
}


def packet(step: str = "identify") -> harness.Packet:
    return harness.Packet("p", step, {}, [], [], words=WORDS)


def score(expect: dict, route: str, control: str = "none", answer: str = ""):
    return harness.score_utterance(
        packet(),
        expect,
        route=route,
        control=control,
        answer=answer,
        guide="tea",
    )


def test_a_question_about_the_next_step_is_answered_not_acted_on() -> None:
    expect = {"route": "conversation", "answer": {"mentions": ["{next}"]}}
    ok = score(expect, "conversation", answer="Next, fill the kettle.")
    assert ok["ok"]
    acted = score(expect, "tea", control="skip")
    assert acted["unintended"] and not acted["ok"]
    vague = score(expect, "conversation", answer="Keep going!")
    assert vague["problems"] == ["answer mentions none of ['fill', 'water']"]


def test_controls_missed_and_allowed_alternatives() -> None:
    assert score({"control": "skip"}, "tea", control="skip")["ok"]
    missed = score({"control": "skip"}, "conversation")
    assert missed["missed"] and not missed["ok"]
    either = {"control": ["stop", "none"]}
    assert score(either, "tea", control="stop")["ok"]
    assert score(either, "conversation")["ok"]


def test_answers_must_not_claim_what_the_state_does_not_say() -> None:
    expect = {"route": "conversation", "answer": {"not": ["seconds left"]}}
    bad = score(expect, "conversation", answer="About 30 seconds left.")
    assert bad["problems"] == ["answer says 'seconds left'"]


@pytest.mark.parametrize(
    ("labels", "expect", "ok"),
    [
        (["negative", "positive", "unknown", "positive"], "done", True),
        (["positive", "negative", "negative", "negative"], "done", False),
        (["positive", "unknown", "positive"], "not_done", False),
        (["positive", "positive"], "either", True),
    ],
)
def test_the_live_rule_decides_a_frame_group(labels, expect, ok) -> None:
    fired = harness.fires(labels, required=2, window=4)
    assert harness.group_verdict(expect, fired, labels) is ok


def test_asking_is_scored_against_the_group() -> None:
    must = {"asks": True, "asks_about": ["unit"]}
    assert harness.ask_verdict(must, "Which unit is the display in?") == []
    assert harness.ask_verdict(must, None) == ["did not ask when stuck"]
    assert harness.ask_verdict(must, "Is the water hot?") == [
        "question is not about any of ['unit']"
    ]
    assert harness.ask_verdict({"asks": False}, "Which unit?") == [
        "asked: 'Which unit?'"
    ]


def frames_index(tmp_path: Path, seconds: int = 60) -> Path:
    rows = ["source_time_s,frame_image"]
    for t in range(seconds):
        name = f"f{t:03d}.jpg"
        Image.new("RGB", (32, 24), "gray").save(tmp_path / name)
        rows.append(f"{t},{name}")
    (tmp_path / "index.csv").write_text("\n".join(rows) + "\n")
    return tmp_path / "index.csv"


def test_a_stuck_heating_check_says_what_it_would_ask(tmp_path: Path) -> None:
    out = tmp_path / "out"
    here = Path(__file__).resolve().parent
    harness.main(
        [
            "--only",
            "heat_water",
            "--frames-index",
            str(frames_index(tmp_path)),
            "--fakes",
            str(here / "smoke.fakes.yaml"),
            "--out",
            str(out),
        ]
    )
    results = json.loads((out / "results.json").read_text())
    by_name = {r["packet"]: r for r in results}
    assert set(by_name) == {"heat_water", "heat_water.asking"}
    (stuck,) = [
        g
        for g in by_name["heat_water"]["frames"]
        if g["name"] == "rising_unit_unknown"
    ]
    assert stuck["question"] == "Does the display show Celsius or Fahrenheit?"
    assert stuck["counts"] == {"unknown": len(stuck["frames"])}  # no unit


def test_a_plumbing_run_writes_a_report(tmp_path: Path) -> None:
    rows = ["source_time_s,frame_image"]
    for t in range(20):
        name = f"f{t:03d}.jpg"
        Image.new("RGB", (32, 24), "gray").save(tmp_path / name)
        rows.append(f"{t},{name}")
    (tmp_path / "index.csv").write_text("\n".join(rows) + "\n")
    out = tmp_path / "out"
    here = Path(__file__).resolve().parent
    harness.main(
        [
            "--only",
            "identify",
            "--frames-index",
            str(tmp_path / "index.csv"),
            "--fakes",
            str(here / "smoke.fakes.yaml"),
            "--out",
            str(out),
        ]
    )
    (result,) = json.loads((out / "results.json").read_text())
    assert result["packet"] == "identify"
    assert {g["name"] for g in result["frames"]} >= {"label_held_up"}
    said = {u["say"] for u in result["utterances"]}
    assert "What's the next step?" in said  # common.yaml was added
    assert "| Frame groups |" in (out / "report.md").read_text()
