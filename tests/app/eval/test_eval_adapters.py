# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each adapter converts a recorded output of its harness (format taken from the harness source)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import eval_support
import pytest
from xr_ai_eval import AdapterContext, find_adapter, load_profile, read_records
from xr_ai_eval.adapters import adapter_names
from xr_ai_eval.cli import main
from xr_ai_eval_adapters import ADAPTERS, harnesses
from xr_ai_eval_adapters.base import HarnessRun, Line, run_streaming

# Fixtures live in eval_support (a conftest.py here would shadow tests/conftest.py).
stub = eval_support.stub
write_profile = eval_support.write_profile

FIXTURES = Path(__file__).parent / "fixtures" / "harness"
REPO = Path(__file__).resolve().parents[3]
STEP_S = 2.0  # recorded outputs get one result line every two seconds


def recorded(name: str, step_s: float = STEP_S, code: int = 1) -> HarnessRun:
    texts = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    lines = [Line(text, (i + 1) * step_s) for i, text in enumerate(texts)]
    return HarnessRun(code, lines, (len(texts) + 1) * step_s)


def by_case(parsed):
    return {p.case: p for p in parsed}


def test_all_six_adapters_are_registered_as_entry_points():
    expected = {
        "tea-making",
        "lab-instrument-monitoring",
        "conversation-example",
        "conversation-context",
        "xr-render-demo",
        "procedure-capture-guide",
    }
    assert set(ADAPTERS) == expected
    assert expected <= set(adapter_names())
    assert find_adapter("tea-making") is harnesses.TEA_MAKING
    assert find_adapter("absent") is None


def test_tea_lines(tmp_path):
    parsed = by_case(harnesses.parse_tea(recorded("tea.txt"), tmp_path))
    assert set(parsed) == {"idle-conversation", "unreadable-label-commits-no-placeholders", "answer-question"}
    assert parsed["idle-conversation"].passed and parsed["idle-conversation"].score == 1.0
    miss = parsed["unreadable-label-commits-no-placeholders"]
    assert not miss.passed and miss.score == 0.0 and miss.detail["tools"] == "['advance_step']"
    assert miss.latency_ms == pytest.approx(STEP_S * 1000)  # gap between result lines
    assert parsed["idle-conversation"].latency_ms == pytest.approx(STEP_S * 1000)


def test_lab_tools_and_visual(tmp_path):
    tools = by_case(harnesses.parse_lab_tools(recorded("lab_tools.txt"), tmp_path))
    assert tools["reading-temperature"].passed and not tools["idle-chat"].passed
    assert "unknown-tool-case" in tools and len(tools) == 3  # the indented content line is not a case
    visual = by_case(harnesses.parse_lab_visual(recorded("lab_visual.txt"), tmp_path))
    assert visual["monitor-unchanged"].passed and visual["monitor-changed"].detail["response"] == "'Looks the same.'"


def test_conversation_example_prefers_the_jsonl_file(tmp_path):
    shutil.copy(FIXTURES / "conversation_example_records.jsonl", tmp_path / "records.jsonl")
    parsed = by_case(harnesses.parse_conversation_example(recorded("conversation_example.txt"), tmp_path))
    assert parsed["flow-a/turn-1"].passed and parsed["flow-a/turn-1"].score == 1.0
    partial = parsed["flow-a/turn-2"]
    assert not partial.passed and partial.score == 0.5 and partial.detail["checks"]["answered"] is False
    assert parsed["flow-b/turn-1"].error == "RuntimeError: endpoint down"


def test_conversation_example_falls_back_to_printed_lines(tmp_path):
    parsed = by_case(harnesses.parse_conversation_example(recorded("conversation_example.txt"), tmp_path))
    assert [p.passed for p in parsed.values()] == [True, False, True]


def test_conversation_context_conflict_and_compaction(tmp_path):
    conflicts = by_case(harnesses.parse_conflicts(recorded("conflicts.txt"), tmp_path))
    assert conflicts["corrected-meeting-day"].passed and conflicts["corrected-meeting-day"].score == 1.0
    assert conflicts["meeting-room"].score == pytest.approx(0.4) and not conflicts["meeting-room"].passed
    assert conflicts["allergy"].error.startswith("seeding failed")
    probes = by_case(harnesses.parse_compaction(recorded("compaction.txt"), tmp_path))
    assert set(probes) == {"lab/user-name", "lab/lamp-serial", "renovation/allergy"}
    assert probes["lab/lamp-serial"].detail["in_prompt"] is False and probes["lab/lamp-serial"].score == 0.2
    assert probes["lab/user-name"].latency_ms == pytest.approx(STEP_S * 1000)  # measured from the turn line


def test_render_pads_status_and_keeps_long_fail_reasons(tmp_path):
    parsed = by_case(harnesses.parse_render(recorded("render.txt"), tmp_path))
    assert set(parsed) == {"make_red_sphere", "place_two_cubes", "clear-scene"}
    assert parsed["make_red_sphere"].passed
    long_fail = parsed["clear-scene"]
    assert not long_fail.passed and "forbidden=['delete_all']" in long_fail.detail["failure"]
    assert long_fail.detail["response"] == "Deleted."
    assert parsed["place_two_cubes"].detail["failure"].startswith("missing=['add_primitive']")


def test_procedure_capture_visual_rows_and_captions(tmp_path):
    (tmp_path / "out").mkdir()
    shutil.copy(FIXTURES / "visual_results.jsonl", tmp_path / "out" / "results.jsonl")
    shutil.copy(FIXTURES / "caption_initial.json", tmp_path / "out" / "initial.json")
    shutil.copy(FIXTURES / "caption_move_right.json", tmp_path / "out" / "move_right.json")
    (tmp_path / "out" / "prompt.txt").write_text("not json")
    run = HarnessRun(0, [], 1.0)
    visual = by_case(harnesses.parse_visual_checks(run, tmp_path))
    good = visual["kettle_boiling/boil/trial1"]
    assert good.passed and good.score == 1.0 and good.latency_ms == pytest.approx(810.0)
    assert good.latency_basis == "harness_reported"
    assert visual["kettle_idle/boil/trial1"].score == 0.0
    assert visual["cup_empty/pour/trial1"].error == "VLM unavailable"
    captions = by_case(harnesses.parse_captions(run, tmp_path))
    assert captions["initial"].passed and captions["move_right"].error == "caption JSON invalid"


def _context(tmp_path, profile=None, path=None):
    return AdapterContext(
        repo_root=REPO,
        profile_name=profile.name if profile else "local",
        profile_path=path,
        profile=profile,
        commit="abc12345",
        run_id="r1",
        timestamp="2026-11-04T00:00:00Z",
        work_dir=tmp_path,
    )


def test_run_converts_to_records_and_marks_missing_cases(tmp_path):
    adapter = harnesses.TEA_MAKING.with_executor(lambda argv, cwd, env: recorded("tea.txt"))
    records = adapter.run(_context(tmp_path))
    by_id = {r.case: r for r in records}
    assert by_id["idle-conversation"].suite == "tea-making" and by_id["idle-conversation"].role == "answer"
    assert by_id["idle-conversation"].tokens == {"input": None, "output": None}
    assert by_id["idle-conversation"].detail["latency_basis"] == "gap_between_output_lines"
    assert by_id["idle-conversation"].detail["profile_honored"] is False  # tea hardcodes yaml/models.json
    assert by_id["idle-conversation"].detail["requested_profile"] == "local"
    assert by_id["idle-conversation"].profile == "sample-models"
    missing = [r for r in records if r.error == "harness produced no result for this case"]
    assert missing and all(not r.passed and r.score == 0.0 for r in missing)  # cases.yaml has more cases
    assert all(r.preset != "" for r in records)


def test_run_records_the_profile_and_preset_when_the_harness_takes_it(tmp_path, stub):
    path = write_profile(tmp_path, "local", stub.url)
    profile = load_profile(str(path), [])
    seen = {}

    def executor(argv, cwd, env):
        seen["argv"] = argv
        return recorded("conflicts.txt")

    records = harnesses.CONVERSATION_CONTEXT.with_executor(executor).run(_context(tmp_path, profile, path))
    assert seen["argv"][-4:] == ["--models", str(path), "--model", "llm"]
    assert seen["argv"][:5] == ["uv", "--config-file", "uv.toml", "run", "--project"]
    assert seen["argv"][5:8] == ["agent-sdk/xr-ai-conversation", "--group", "eval"]
    conflict = next(r for r in records if r.case == "conflict/meeting-room")
    assert conflict.profile == "local" and conflict.preset == "big-stub" and conflict.gpu_profile == "dual_48G_ada"
    assert conflict.detail["profile_honored"] is True


def test_a_crashed_harness_is_one_error_record(tmp_path):
    crashed = HarnessRun(3, [Line("Traceback ...", 0.1), Line("ImportError: nope", 0.2)], 0.3)
    adapter = harnesses.XR_RENDER_DEMO.with_executor(lambda *a: crashed)
    records = adapter.run(_context(tmp_path))
    assert len(records) == 1 and records[0].error.startswith("harness exited 3 with no results")
    assert "ImportError: nope" in records[0].error and not records[0].passed


def test_cases_without_timestamps_get_the_run_average(tmp_path):
    shutil.copy(FIXTURES / "conversation_example_records.jsonl", tmp_path / "records.jsonl")
    quiet = HarnessRun(1, [], 9.0)  # harness printed nothing, only wrote the file
    records = harnesses.CONVERSATION_EXAMPLE.with_executor(lambda *a: quiet).run(_context(tmp_path))
    assert {r.latency_ms for r in records} == {3000.0}
    assert {r.detail["latency_basis"] for r in records} == {"run_average"}
    assert next(r for r in records if r.case == "flow-b/turn-1").error


def test_run_streaming_timestamps_each_line():
    code = "import time; print('a'); time.sleep(0.3); print('b')"
    result = run_streaming([sys.executable, "-c", code], Path.cwd(), {})
    assert [line.text for line in result.lines] == ["a", "b"] and result.returncode == 0
    assert result.lines[1].at_s - result.lines[0].at_s >= 0.25


def test_cli_run_converts_through_a_registered_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(harnesses.TEA_MAKING, "_executor", lambda argv, cwd, env: recorded("tea.txt"))
    evals = tmp_path / "evals"
    (evals / "profiles").mkdir(parents=True)
    write_profile(evals / "profiles", "local", "http://unused")
    out = tmp_path / "adapter.jsonl"
    code = main(["run", "--suite", "tea-making", "--profile", "local", "--evals-dir", str(evals), "--out", str(out)])
    assert code == 1  # one MISS and the cases the harness never reported
    records = list(read_records(out))
    assert {r.suite for r in records} == {"tea-making"} and any(r.passed for r in records)
