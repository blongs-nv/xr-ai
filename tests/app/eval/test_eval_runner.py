# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runner against a stub OpenAI-compatible server: schema, scorers, profiles, CLI, exit codes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import eval_support
import pytest
from xr_ai_eval import CaseError, Record, SchemaError, load_profile, load_suite, read_records
from xr_ai_eval.cli import main
from xr_ai_eval.client import ChatClient, chat_url
from xr_ai_eval.profiles import ProfileError
from xr_ai_eval.runner import run_suite, select_cases
from xr_ai_eval.scorers import pick_choice, score_output

# Fixtures live in eval_support (a conftest.py here would shadow tests/conftest.py).
stub = eval_support.stub
evals_dir = eval_support.evals_dir
write_profile = eval_support.write_profile


def _run(suite_dir: Path, profile, **kwargs):
    async def go():
        client = ChatClient(timeout_s=10)
        try:
            return await run_suite(
                load_suite(suite_dir, "kettle"), profile, client, commit="abc12345", run_id="r1", **kwargs
            )
        finally:
            await client.close()

    return asyncio.run(go())


def test_run_suite_writes_schema_records(evals_dir, stub, tmp_path):
    stub.rules = {"kettle": "Positive.", "Extract": '```json\n{"temp": 90, "unit": "F"}\n```'}
    profile = load_profile(str(write_profile(tmp_path, "local", stub.url)), [])
    records = _run(evals_dir, profile, gpu_memory=lambda: 20480.0, now=lambda: "2026-11-04T00:00:00Z")
    by_case = {r.case: r for r in records}
    assert set(by_case) == {"boiling", "json-fields"}  # skip_reason writes no record
    boiling = by_case["boiling"]
    assert boiling.passed and boiling.score == 1.0 and boiling.role == "perceive"
    assert boiling.preset == "cosmos_stub" and boiling.profile == "local"
    assert boiling.tokens == {"input": 10, "output": 3} and boiling.latency_ms > 0
    assert boiling.gpu_profile == "dual_48G_ada" and boiling.gpu_memory_mb == 20480.0
    assert boiling.commit == "abc12345" and boiling.timestamp == "2026-11-04T00:00:00Z"
    fields = by_case["json-fields"]  # role extract -> alias entry llm; partial credit 1 of 2 >= 0.5
    assert fields.score == 0.5 and fields.passed and fields.preset == "big-stub"
    assert fields.detail["wrong_fields"] == ["unit"]
    wire = boiling.to_json()
    assert wire["pass"] is True and wire["schema_version"] == 1 and "passed" not in wire
    assert Record.from_json(json.loads(json.dumps(wire))) == boiling


def test_requests_carry_frames_context_model_and_path(evals_dir, stub, tmp_path):
    profile = load_profile(str(write_profile(tmp_path, "local", stub.url)), [])
    _run(evals_dir, profile)
    vlm_request = next(r for r in stub.requests if r["model"] == "cosmos_stub")
    assert vlm_request["_path"] == "/v1/chat/completions"
    parts = vlm_request["messages"][-1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    llm_request = next(r for r in stub.requests if r["model"] == "big-stub")
    assert llm_request["_path"] == "/v1/chat/completions"  # base_url already ends in /v1
    assert [m["content"] for m in llm_request["messages"]] == ["earlier", "Extract the temp"]


def test_missing_usage_gives_null_tokens(evals_dir, stub, tmp_path):
    stub.usage = None
    stub.rules = {"kettle": "positive"}
    records = _run(evals_dir, load_profile(str(write_profile(tmp_path, "p", stub.url)), []))
    assert records[0].tokens == {"input": None, "output": None}


def test_http_error_is_one_error_record(evals_dir, stub, tmp_path):
    stub.fail_with = 500
    records = _run(evals_dir, load_profile(str(write_profile(tmp_path, "p", stub.url)), []))
    assert len(records) == 2 and all(r.error and not r.passed and r.score == 0.0 for r in records)
    assert "HTTPStatusError" in records[0].error


def test_missing_api_key_env_is_an_error_record(evals_dir, stub, tmp_path, monkeypatch):
    monkeypatch.delenv("STUB_KEY", raising=False)
    profile = load_profile(str(write_profile(tmp_path, "p", stub.url, api_key_env="STUB_KEY")), [])
    records = _run(evals_dir, profile, roles=["extract"])
    assert records[0].error and "STUB_KEY" in records[0].error
    monkeypatch.setenv("STUB_KEY", "secret")
    _run(evals_dir, profile, roles=["extract"])
    assert stub.requests[-1]["_auth"] == "Bearer secret"


def test_selection_by_tags_and_roles(evals_dir):
    suite = load_suite(evals_dir, "kettle")
    assert [c.id for c in select_cases(suite)] == ["boiling", "json-fields"]
    assert [c.id for c in select_cases(suite, tags=["visual"])] == ["boiling"]
    assert [c.id for c in select_cases(suite, roles=["extract"])] == ["json-fields"]


def test_judge_scorer_records_role_and_preset(evals_dir, stub, tmp_path):
    (evals_dir / "kettle" / "cases" / "judge.yaml").write_text(
        """
suite: kettle
version: 1
cases:
  - id: judged
    role: perceive
    input: {text: "Describe the kettle"}
    scorer: {type: judge, role: extract, rubric: "mentions steam"}
    pass_threshold: 0.7
""",
        encoding="utf-8",
    )
    stub.rules = {"Rubric": "0.8", "Describe": "steam rises"}
    records = _run(evals_dir, load_profile(str(write_profile(tmp_path, "p", stub.url)), []), tags=[])
    judged = next(r for r in records if r.case == "judged")
    assert judged.passed and judged.score == 0.8
    assert judged.detail["judge_role"] == "extract" and judged.detail["judge_preset"] == "big-stub"


@pytest.mark.parametrize(
    ("scorer", "output", "expected"),
    [
        ({"type": "exact", "expect": "ok"}, "  ok \n", 1.0),
        ({"type": "exact", "expect": "ok"}, "ok!", 0.0),
        ({"type": "contains", "expect": ["a", "b"]}, "a and b", 1.0),
        ({"type": "contains", "expect": ["a", "b"]}, "only a", 0.0),
        ({"type": "regex", "expect": r"\d{2}C"}, "about 90C now", 1.0),
        ({"type": "choice", "expect": "yes", "choices": ["yes", "no"]}, "Yes.", 1.0),
        ({"type": "choice", "expect": "yes", "choices": ["yes", "no"]}, "yes or no", 0.0),
        ({"type": "json_match", "expect": {"a": 1, "b": 2}}, 'x {"a": 1, "b": 3} y', 0.5),
        ({"type": "json_match", "expect": {"a": 1}}, "not json", 0.0),
    ],
)
def test_scorers(scorer, output, expected):
    score, detail = asyncio.run(score_output(scorer, output))
    assert score == expected and detail["output"] == output


def test_pick_choice_requires_whole_word():
    assert pick_choice("the answer is not-positive", ["positive", "negative"]) is None
    assert pick_choice("negative", ["positive", "negative"]) == "negative"


def test_chat_url_joins_with_or_without_v1():
    assert chat_url("http://h:1") == "http://h:1/v1/chat/completions"
    assert chat_url("http://h:1/v1/") == "http://h:1/v1/chat/completions"


def _doc(case: str, suite: str = "kettle") -> str:
    return f"suite: {suite}\nversion: 1\ncases: [{{{case}}}]"


_EXACT = "scorer: {type: exact, expect: x}"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_doc(f"id: a, role: r, input: {{text: t}}, {_EXACT}", suite="wrong"), "directory name"),
        (_doc(f"id: a, input: {{text: t}}, {_EXACT}"), "'role' is required"),
        (_doc("id: a, role: r, input: {text: t}, scorer: {type: bogus}"), "scorer.type"),
        (
            _doc("id: a, role: r, input: {text: t}, scorer: {type: choice, expect: z, choices: [y]}"),
            "not one of",
        ),
        (_doc(f"id: a, role: r, input: {{text: t, frames: [nope.jpg]}}, {_EXACT}"), "not found"),
        (_doc(f"id: a, role: r, input: {{text: t}}, {_EXACT}, pass_threshold: 2"), "pass_threshold"),
    ],
)
def test_invalid_cases_raise(tmp_path, text, message):
    (tmp_path / "kettle" / "cases").mkdir(parents=True)
    (tmp_path / "kettle" / "cases" / "a.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(CaseError, match=message):
        load_suite(tmp_path, "kettle")


def test_duplicate_ids_raise(evals_dir):
    (evals_dir / "kettle" / "cases" / "dup.yaml").write_text(
        _doc(f"id: boiling, role: r, input: {{text: t}}, {_EXACT}"),
        encoding="utf-8",
    )
    with pytest.raises(CaseError, match="duplicate"):
        load_suite(evals_dir, "kettle")


def test_profile_resolution_and_errors(tmp_path, stub):
    write_profile(tmp_path, "hybrid", stub.url)
    assert load_profile("hybrid", [tmp_path]).gpu_profile == "dual_48G_ada"
    with pytest.raises(ProfileError, match="looked for"):
        load_profile("absent", [tmp_path])
    legacy = tmp_path / "models.json"  # unwrapped legacy form, modality-named entries
    legacy.write_text(json.dumps({"vlm": {"adapter": {"preset": "p"}, "endpoint": {"base_url": "http://x"}}}))
    assert load_profile(str(legacy), []).endpoint("perceive").preset == "p"
    with pytest.raises(ProfileError, match="no model for role"):
        load_profile(str(legacy), []).endpoint("answer")


def test_readers_ignore_unknown_fields_and_reject_bad_lines(tmp_path):
    good = Record("s", "c", "r", "m", "p", True, 1.0, 1.0, {"input": 1, "output": 2}, None, "abc").to_json()
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({**good, "future_field": [1]}) + "\n\n", encoding="utf-8")
    assert [r.case for r in read_records(path)] == ["c"]
    path.write_text(json.dumps({**good, "score": 1.5}) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match=r"r.jsonl:1"):
        list(read_records(path))


def test_cli_run_exit_codes_and_append(evals_dir, stub, tmp_path, capsys):
    (evals_dir / "profiles").mkdir()
    write_profile(evals_dir / "profiles", "local", stub.url)
    out = tmp_path / "out" / "r.jsonl"
    base = ["run", "--suite", "kettle", "--profile", "local", "--evals-dir", str(evals_dir)]
    base += ["--no-gpu-memory", "--out", str(out)]
    stub.rules = {"kettle": "positive", "Extract": '{"temp": 90, "unit": "C"}'}
    assert main(base) == 0
    stub.rules = {"kettle": "negative"}
    assert main(base) == 1  # appended, with one failure
    assert len(list(read_records(out))) == 4
    assert main(["run", "--suite", "nope", "--profile", "local", "--evals-dir", str(evals_dir)]) == 2
    assert main(["run", "--suite", "kettle", "--profile", "missing", "--evals-dir", str(evals_dir)]) == 2
    assert "error:" in capsys.readouterr().err


# Copied from app-sdk/xr-ai-app/docs/PROFILES_SCHEMA.md ("Profile file format"), so the runner is
# tested on the documented example; update both together.
SCHEMA_EXAMPLE = """
profile:
  name: hybrid
  description: answer and extract on a hosted large model; perception and decisions local
  gpu_profile: dual_48G_ada
models:
  perceive:
    adapter: {preset: cosmos3_nano_reasoner}
    endpoint: {base_url: "http://localhost:8100"}
    deployment: {ownership: managed, service: vlm-server}
  decide:
    adapter: {kind: systemone}
    endpoint: {base_url: "http://localhost:8110"}
    deployment: {ownership: managed, service: clef-server}
  answer:
    adapter: {kind: openai_compat, model_name: "nvidia/llama-3.3-nemotron-super-49b-v1.5"}
    endpoint: {base_url: "https://integrate.api.nvidia.com/v1", api_key_env: NVIDIA_API_KEY}
  speech_out:
    adapter: {preset: pocket_tts}
    endpoint: {base_url: "http://localhost:8105"}
    deployment: {ownership: managed, service: pocket-tts}
roles:
  extract: answer
  ack: answer
budgets:
  perceive: {max_concurrency: 2}
"""


def test_resolver_loads_the_profiles_schema_example(tmp_path):
    path = tmp_path / "models.hybrid.yaml"
    path.write_text(SCHEMA_EXAMPLE, encoding="utf-8")
    profile = load_profile("hybrid", [tmp_path])
    assert profile.name == "hybrid" and profile.gpu_profile == "dual_48G_ada"
    assert profile.endpoint("perceive").preset == "cosmos3_nano_reasoner"
    assert profile.endpoint("perceive").base_url == "http://localhost:8100"
    answer = profile.endpoint("answer")
    assert answer.model == "nvidia/llama-3.3-nemotron-super-49b-v1.5" and answer.api_key_env == "NVIDIA_API_KEY"
    assert profile.entry_key("extract") == "answer" and profile.entry_key("ack") == "answer"
    # The Clef `decide` entry resolves, but the runner speaks only OpenAI chat; the
    # conversation-example adapter covers Clef.
    assert profile.endpoint("decide").base_url == "http://localhost:8110"
