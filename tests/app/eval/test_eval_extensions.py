# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner schema extensions (gap theme M): system, turns, audio, driver, python scorer."""

from __future__ import annotations

import asyncio
from pathlib import Path

import eval_support
import pytest
from xr_ai_eval import CaseError, load_profile, load_suite
from xr_ai_eval.client import ChatClient, transcription_url
from xr_ai_eval.runner import run_suite

stub = eval_support.stub
write_profile = eval_support.write_profile

SCORERS = """
def exact_words(case, result):
    expected = case.scorer["expect"].split()
    got = result.text.split()
    score = sum(a == b for a, b in zip(expected, got)) / len(expected)
    return score, {"n": len(got), "replies": list(result.replies)}

def broken(case, result):
    raise KeyError("boom")

def drive(endpoint, case):
    return "driver saw " + endpoint.base_url.split(":")[0] + " " + case.id
"""


def _suite(tmp_path: Path, cases: str) -> Path:
    root = tmp_path / "evals"
    (root / "s" / "cases").mkdir(parents=True)
    (root / "s" / "cases" / "c.yaml").write_text(
        "suite: s\nversion: 1\ndefaults: {role: speech_in}\ncases:\n" + cases, encoding="utf-8"
    )
    (root / "s" / "cases" / "hooks.py").write_text(SCORERS, encoding="utf-8")
    (root / "s" / "cases" / "tone.wav").write_bytes(b"RIFF....WAVE")
    return root


def _run(root: Path, profile):
    async def go():
        client = ChatClient(timeout_s=10)
        try:
            return await run_suite(load_suite(root, "s"), profile, client, commit="c0", run_id="r")
        finally:
            await client.close()

    return asyncio.run(go())


def _profile(tmp_path, stub):
    path = eval_support.write_profile(tmp_path, "t", stub.url)
    return load_profile(str(path), [])


def test_system_and_turns_are_sent_in_order(stub, tmp_path):
    root = _suite(
        tmp_path,
        """
  - id: flow
    role: perceive
    system: You are a kettle monitor.
    input:
      context: [{role: user, text: earlier}]
      turns: [{text: "first q"}, {text: "second q"}]
    scorer: {type: python, function: "hooks.py:exact_words", expect: "unknown unknown"}
""",
    )
    stub.rules = {"first": "alpha", "second": "beta"}
    (record,) = _run(root, _profile(tmp_path, stub))
    assert record.error is None and record.score == 0.0
    assert record.detail["replies"] == ["alpha", "beta"] and record.detail["output"] == "beta"
    assert record.tokens == {"input": 20, "output": 6}
    first, second = stub.requests
    assert [m["role"] for m in first["messages"]] == ["system", "user", "user"]
    assert first["messages"][0]["content"] == "You are a kettle monitor."
    assert [m["role"] for m in second["messages"]] == ["system", "user", "user", "assistant", "user"]
    assert second["messages"][3]["content"] == "alpha"


def test_audio_case_posts_to_transcriptions(stub, tmp_path):
    root = _suite(
        tmp_path,
        """
  - id: stt
    input: {audio: tone.wav}
    scorer: {type: python, function: "hooks.py:exact_words", expect: "hello world"}
""",
    )
    stub.transcript = "hello world"
    record = _run(root, _profile(tmp_path, stub))[0]
    assert record.passed and record.score == 1.0 and record.tokens == {"input": None, "output": None}
    assert stub.uploads[0]["content_type"].startswith("multipart/form-data") and b"RIFF" in stub.uploads[0]["raw"]
    assert transcription_url("http://x/v1") == "http://x/v1/audio/transcriptions"
    assert transcription_url("http://x") == "http://x/v1/audio/transcriptions"


def test_driver_replaces_model_call_and_errors_become_records(stub, tmp_path):
    root = _suite(
        tmp_path,
        """
  - id: drv
    input: {driver: "hooks.py:drive"}
    scorer: {type: contains, expect: "driver saw http drv"}
  - id: bad-scorer
    input: {text: hi}
    scorer: {type: python, function: "hooks.py:broken"}
""",
    )
    ok, bad = _run(root, _profile(tmp_path, stub))
    assert ok.passed and ok.latency_ms == 0.0
    assert not bad.passed and "KeyError" in bad.error


def test_python_scorer_and_driver_files_must_exist_unless_skipped(tmp_path):
    root = _suite(
        tmp_path,
        """
  - id: x
    input: {text: hi}
    scorer: {type: python, function: "nope.py:f"}
""",
    )
    with pytest.raises(CaseError, match="hook file not found"):
        load_suite(root, "s")
    (root / "s" / "cases" / "c.yaml").write_text(
        """suite: s
version: 1
defaults: {role: perceive}
cases:
  - id: pending
    skip_reason: not recorded yet
    input: {audio: missing.wav, frames: [missing.jpg]}
    scorer: {type: exact, expect: a}
  - id: live
    input: {audio: missing.wav}
    scorer: {type: exact, expect: a}
""",
        encoding="utf-8",
    )
    with pytest.raises(CaseError, match="audio 'missing.wav' not found"):
        load_suite(root, "s")


def test_hooks_cannot_escape_the_suite_directory(tmp_path):
    scorer = '{type: python, function: "../../outside.py:f"}'
    root = _suite(tmp_path, f"  - id: x\n    input: {{text: hi}}\n    scorer: {scorer}\n")
    (tmp_path / "evals" / "outside.py").write_text("def f(case, result):\n    return 1\n")
    with pytest.raises(CaseError, match="outside the suite directory"):
        load_suite(root, "s")


@pytest.mark.parametrize(
    "block, message",
    [
        ("input: {turns: [{text: a}], text: b}\n    scorer: {type: exact, expect: a}", "belong to each turn"),
        ("input: {text: a}\n    scorer: {type: python, function: nofunc}", "file.py:function"),
        ("input: {text: a}\n    system: [1]\n    scorer: {type: exact, expect: a}", "'system' must be a string"),
        ("input: {turns: [{}]}\n    scorer: {type: exact, expect: a}", "each turn needs"),
    ],
)
def test_invalid_extension_blocks(tmp_path, block, message):
    root = _suite(tmp_path, f"  - id: x\n    role: perceive\n    {block}\n")
    with pytest.raises(CaseError, match=message):
        load_suite(root, "s")


def test_params_reach_driver_and_scorer(stub, tmp_path):
    root = _suite(
        tmp_path,
        """
  - id: p
    input: {driver: "hooks.py:echo_params", params: {k: [1, 2]}}
    scorer: {type: contains, expect: "[1, 2]"}
""",
    )
    hooks = "def echo_params(endpoint, case):\n    return str(case.params['k'])\n"
    (root / "s" / "cases" / "hooks.py").write_text(hooks)
    (record,) = _run(root, _profile(tmp_path, stub))
    assert record.passed
