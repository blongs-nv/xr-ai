# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix runs, the scorecard report against a fixture, and the capture helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import eval_support
from xr_ai_eval import metrics, read_records
from xr_ai_eval.cli import main
from xr_ai_eval.report import build_scorecard

# Fixtures live in eval_support (a conftest.py here would shadow tests/conftest.py).
stub = eval_support.stub
evals_dir = eval_support.evals_dir
write_profile = eval_support.write_profile

FIXTURES = Path(__file__).parent / "fixtures"


def test_report_matches_fixture(tmp_path):
    out = tmp_path / "scorecard.md"
    assert main(["report", "--results", str(FIXTURES / "scorecard_results.jsonl"), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == (FIXTURES / "scorecard_expected.txt").read_text(encoding="utf-8")


def test_report_is_order_independent():
    records = list(read_records(FIXTURES / "scorecard_results.jsonl"))
    assert build_scorecard(records) == build_scorecard(reversed(records))


def test_report_rejects_bad_results(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"suite": "s"}\n', encoding="utf-8")
    assert main(["report", "--results", str(bad), "--out", str(tmp_path / "o.md")]) == 2
    assert "bad.jsonl:1" in capsys.readouterr().err


def test_matrix_runs_every_profile_and_preset(evals_dir, stub, tmp_path):
    profiles = evals_dir / "profiles"
    profiles.mkdir()
    write_profile(profiles, "local", stub.url)
    write_profile(profiles, "hybrid", stub.url)
    stub.rules = {"kettle": "positive", "Extract": '{"temp": 90, "unit": "C"}'}
    out = tmp_path / "matrix.jsonl"
    args = ["matrix", "--suite", "kettle", "--profiles", "local,hybrid", "--presets", "small,large"]
    code = main([*args, "--evals-dir", str(evals_dir), "--no-gpu-memory", "--out", str(out)])
    assert code == 0
    records = list(read_records(out))
    assert len(records) == 8  # 2 profiles x 2 presets x 2 executed cases
    assert {(r.profile, r.preset) for r in records} == {(p, m) for p in ("local", "hybrid") for m in ("small", "large")}
    assert {r["model"] for r in stub.requests} == {"small", "large"}  # the preset is the requested model
    scorecard = build_scorecard(records)
    assert "| kettle | perceive | large | hybrid, local | 2 | 100.0% |" in scorecard


def test_matrix_without_presets_uses_each_profile_model(evals_dir, stub, tmp_path):
    (evals_dir / "profiles").mkdir()
    write_profile(evals_dir / "profiles", "local", stub.url)
    out = tmp_path / "m.jsonl"
    stub.rules = {"kettle": "negative"}
    args = ["matrix", "--suite", "kettle", "--profiles", "local", "--evals-dir", str(evals_dir)]
    assert main([*args, "--no-gpu-memory", "--out", str(out)]) == 1
    assert {r.preset for r in read_records(out)} == {"cosmos_stub", "big-stub"}


def test_gpu_memory_sums_all_gpus(monkeypatch):
    monkeypatch.setattr(metrics.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    result = subprocess.CompletedProcess([], 0, stdout="20279\n3741\n")
    monkeypatch.setattr(metrics.subprocess, "run", lambda *a, **k: result)
    assert metrics.nvidia_smi_memory_mb() == 24020.0


def test_gpu_memory_is_none_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(metrics.shutil, "which", lambda name: None)
    assert metrics.nvidia_smi_memory_mb() is None


def test_gpu_memory_is_none_when_the_tool_fails(monkeypatch):
    monkeypatch.setattr(metrics.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    monkeypatch.setattr(metrics.subprocess, "run", boom)
    assert metrics.nvidia_smi_memory_mb() is None


def test_git_commit_outside_a_repo_is_unknown(tmp_path):
    assert metrics.git_commit(tmp_path) == "unknown"
