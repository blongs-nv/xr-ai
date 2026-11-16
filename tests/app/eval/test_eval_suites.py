# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tea_visual and speech suites through the runner on CPU stubs.

Plumbing only: no accuracy assertions.
"""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import eval_support
import pytest
from xr_ai_eval import load_suite, read_records
from xr_ai_eval.cli import main

stub = eval_support.stub
SUITES = Path(__file__).resolve().parents[3] / "evals" / "suites"
TEA = SUITES / "tea_visual"
SPEECH = SUITES / "speech"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(suite: str, profile: Path, out: Path, *extra: str) -> int:
    return main(
        ["run", "--suite", suite, "--evals-dir", str(SUITES), "--profile", str(profile), "--out", str(out),
         "--no-gpu-memory", *extra]
    )  # fmt: skip


def test_tea_visual_covers_every_required_category():
    suite = load_suite(SUITES, "tea_visual")
    tags = {tag for case in suite.cases for tag in case.tags}
    required = {"below-target", "at-target", "fahrenheit", "unreadable-units", "empty-vessel", "occluded", "packaging"}
    assert required <= tags
    live = [c for c in suite.cases if c.skip_reason is None]
    assert live and all(f.is_file() for c in live for f in c.frames)
    recorded = [c for c in suite.cases if "recorded" in c.tags]
    assert {t for c in recorded for t in c.tags} >= required
    assert all(c.skip_reason for c in recorded)


def test_tea_visual_runs_and_scorecard_has_rows(stub, tmp_path):
    stub.default = "UNKNOWN"
    profile = eval_support.write_profile(tmp_path, "local", stub.url)
    out = tmp_path / "r.jsonl"
    assert _run("tea_visual", profile, out) == 1  # a constant UNKNOWN answer fails the other cases
    records = list(read_records(out))
    live = [c for c in load_suite(SUITES, "tea_visual").cases if c.skip_reason is None]
    assert len(records) == len(live) and all(r.role == "perceive" and r.error is None for r in records)
    assert any(r.passed for r in records)  # packet-covered, packet-artwork-only, ... expect UNKNOWN
    request = stub.requests[0]["messages"][-1]["content"]
    assert request[1]["image_url"]["url"].startswith("data:image/png;base64,")
    card = tmp_path / "card.md"
    assert main(["report", "--results", str(out), "--out", str(card)]) == 0
    assert "tea_visual" in card.read_text()


def test_speech_suite_loads_with_pending_recordings_skipped():
    suite = load_suite(SUITES, "speech")
    assert {t for c in suite.cases for t in c.tags} >= {"stt", "gating", "recorded", "synthetic"}
    assert all(c.skip_reason for c in suite.cases if "recorded" in c.tags)
    assert not any(c.skip_reason for c in suite.cases if "synthetic" in c.tags)


def test_stt_suite_runs_against_transcription_stub(stub, tmp_path):
    stub.transcript = ""
    profile = eval_support.write_profile(tmp_path, "local", stub.url)
    out = tmp_path / "r.jsonl"
    assert _run("speech", profile, out, "--tags", "stt") == 0
    (record,) = list(read_records(out))
    assert record.case == "stt-silence-empty" and record.passed and record.role == "speech_in"
    assert record.detail["wer"] == 0.0
    stub.transcript = "phantom words"
    assert _run("speech", profile, tmp_path / "r2.jsonl", "--tags", "stt") == 1


def test_wer_scorer():
    scorers = _load(SPEECH / "scorers.py", "speech_scorers")
    case = SimpleNamespace(scorer={"reference": "Hey, let's start the tea"})
    score, detail = scorers.wer(case, SimpleNamespace(text="hey lets start tea now"))
    assert detail["substitutions"] + detail["deletions"] + detail["insertions"] == 2
    assert round(score, 4) == round(1 - 2 / 5, 4)
    assert scorers.wer(case, SimpleNamespace(text="hey lets start the tea"))[0] == 1.0
    assert scorers.wer(SimpleNamespace(scorer={"reference": ""}), SimpleNamespace(text="x"))[0] == 0.0


def test_gating_scorer():
    scorers = _load(SPEECH / "scorers.py", "speech_scorers_g")

    def run(summary, **scorer):
        return scorers.gating(SimpleNamespace(scorer=scorer), SimpleNamespace(text=json.dumps(summary)))[0]

    ok = {"enrolled": True, "transcripts": ["How hot is it"], "event_kinds": []}
    assert run(ok, expect_text="how hot is it") == 1.0
    assert run({**ok, "enrolled": False}, expect_text="how hot is it") == 0.0
    assert run(ok, forbid=["hot is"]) == 0.0
    assert run({"enrolled": False, "transcripts": []}, expect_enrolled=False) == 1.0


def _fake_ffmpeg(tmp_path: Path, frames: int) -> Path:
    path = tmp_path / "ffmpeg"
    path.write_text(
        f"#!{sys.executable}\nimport sys\nargs = sys.argv[1:]\n"
        f"if '-progress' in args:\n    print('frame={frames}')\n"
        "else:\n    open(args[-1], 'wb').write(b'jpeg')\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _bundle(tmp_path: Path, count: int) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "video").mkdir(parents=True)
    (bundle / "video" / "session.264").write_bytes(b"\x00\x00\x00\x01")
    rows = [
        {"frame_id": i + 1, "pts_us": 1000 + i * 500_000, "relative_us": i * 500_000, "track_id": "cam0"}
        for i in range(count)
    ]
    (bundle / "video" / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return bundle


def test_extract_frames_selects_by_time_and_writes_provenance(tmp_path):
    extract = _load(TEA / "extract_frames.py", "tea_extract")
    bundle, out = _bundle(tmp_path, 10), tmp_path / "out"
    ffmpeg = _fake_ffmpeg(tmp_path, 10)
    assert extract.main([str(bundle), "--at", "2.1", "--name", "kettle_at_target", "--frame-id", "3",
                         "--name", "vessel_empty", "--out", str(out), "--ffmpeg", str(ffmpeg)]) == 0  # fmt: skip
    assert (out / "kettle_at_target.jpg").read_bytes() == b"jpeg"
    first = json.loads((out / "kettle_at_target.json").read_text())
    assert first["frame_id"] == 5 and first["decode_position"] == 4 and first["track_id"] == "cam0"
    assert json.loads((out / "vessel_empty.json").read_text())["frame_id"] == 3
    command = extract.ffmpeg_command("ffmpeg", Path("s.264"), 4, Path("o.jpg"))
    assert "select=eq(n\\,4)" in command


def test_extract_frames_refuses_count_mismatch_and_missing_ffmpeg(tmp_path, capsys):
    extract = _load(TEA / "extract_frames.py", "tea_extract2")
    bundle = _bundle(tmp_path, 10)
    args = [str(bundle), "--at", "1", "--name", "x", "--out", str(tmp_path / "o")]
    assert extract.main([*args, "--ffmpeg", str(_fake_ffmpeg(tmp_path, 9))]) == 1
    assert "refusing to label frames" in capsys.readouterr().err
    assert extract.main([*args, "--ffmpeg", "no-such-ffmpeg-binary"]) == 2
    assert "install ffmpeg" in capsys.readouterr().err
    with pytest.raises(extract.ExtractError, match="not in the index"):
        extract.select(extract.read_index(bundle), frame_id=99)


def test_synthetic_frames_render_from_scene_list(tmp_path):
    pytest.importorskip("PIL")
    render = _load(TEA / "render_synthetic.py", "tea_render")
    paths = render.render_all(TEA / "fixtures" / "scenes.yaml", tmp_path)
    committed = sorted(p.name for p in (TEA / "fixtures" / "synthetic").glob("*.png"))
    assert sorted(p.name for p in paths) == committed
