# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sample's two evaluations: their scoring rules, and a plumbing run of
each on scripted replies."""

import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_dialogue  # noqa: E402
import run_steps  # noqa: E402

WORDS = {
    "identify": ["label"],
    "fill_water": ["fill", "water"],
    "done": ["done"],
}


def score(expect: dict, route: str, control: str = "none", answer: str = ""):
    packet = run_dialogue.Packet("p", "identify", {}, [], words=WORDS)
    return run_dialogue.score_utterance(
        packet,
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


def test_a_dialogue_plumbing_run_writes_a_report(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_dialogue.main(
        ["--only", "identify", "--fakes", str(HERE / "smoke.fakes.yaml")]
        + ["--out", str(out)]
    )
    (result,) = json.loads((out / "results.json").read_text())
    said = {u["say"] for u in result["utterances"]}
    assert "What's the next step?" in said  # common.yaml was added
    assert "utterances passed" in (out / "report.md").read_text()


def gray_recording(tmp_path: Path, seconds: int) -> Path:
    rows = ["source_time_s,frame_image"]
    Image.new("RGB", (32, 24), "gray").save(tmp_path / "gray.jpg")
    rows += [f"{t},gray.jpg" for t in range(seconds)]
    (tmp_path / "index.csv").write_text("\n".join(rows) + "\n")
    return tmp_path / "index.csv"


def test_a_step_plumbing_run_runs_every_case_of_the_real_steps(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    run_steps.main(
        ["--frames-index", str(gray_recording(tmp_path, 400))]
        + ["--fakes", str(HERE / "smoke.fakes.yaml"), "--out", str(out)]
    )
    results = json.loads((out / "results.json").read_text())
    cases = {f.stem: f for f in (HERE / "steps").glob("*.yaml")}
    assert {r["step"] for r in results} == set(cases)
    by_name = {(r["step"], r["case"]): r for r in results}
    # The scripted reasoning asks about the unit; the person answers.
    asked = by_name[("heat_water", "unit_unknown_then_answered")]["lines"]
    assert asked[0]["kind"] == "ask" and asked[0]["answer"] == "Fahrenheit"
    assert "cases passed" in (out / "report.md").read_text()
