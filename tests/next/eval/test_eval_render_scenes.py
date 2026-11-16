# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The xr-render scenario suite through the runner on a scripted OpenAI-compatible stub (no accuracy)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from xr_ai_eval import load_suite, read_records
from xr_ai_eval.cli import main

SUITES = Path(__file__).resolve().parents[3] / "evals" / "suites"
SUITE = SUITES / "xr_render_scenes"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"xr_render_scenes_{name}", SUITE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(name: str, **arguments: object) -> dict:
    function = {"name": name, "arguments": json.dumps(arguments)}
    return {"content": None, "tool_calls": [{"id": f"c-{name}", "type": "function", "function": function}]}


class ScriptedLLM:
    """OpenAI-compatible chat stub that replays ``script`` (message dicts) in order, then answers ``ok``."""

    def __init__(self, script: list[dict]) -> None:
        self.script, self.requests = list(script), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                message = outer.script.pop(0) if outer.script else {"content": "ok"}
                message = {"role": "assistant", **message}
                finish = "tool_calls" if message.get("tool_calls") else "stop"
                data = json.dumps({"choices": [{"message": message, "finish_reason": finish}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def llm():
    servers: list[ScriptedLLM] = []

    def make(script: list[dict]) -> ScriptedLLM:
        servers.append(ScriptedLLM(script))
        return servers[-1]

    yield make
    for server in servers:
        server.close()


def _one_case_suite(tmp_path: Path, name: str) -> Path:
    """A copy of the suite holding only ``name``, so a scripted stub answers exactly that case."""
    root = tmp_path / "suites" / "xr_render_scenes"
    (root / "cases").mkdir(parents=True)
    for file in ("drivers.py", "scorers.py"):
        shutil.copy(SUITE / file, root / file)
    import yaml  # noqa: PLC0415

    document = yaml.safe_load((SUITE / "cases" / "scenarios.yaml").read_text(encoding="utf-8"))
    document["cases"] = [c for c in document["cases"] if c["id"] == name]
    (root / "cases" / "scenarios.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return tmp_path / "suites"


def _profile(tmp_path: Path, url: str) -> Path:
    path = tmp_path / "models.local.json"
    doc = {
        "profile": {"name": "local", "gpu_profile": "dual_48G_ada"},
        "models": {"llm": {"adapter": {"kind": "openai_compat", "model_name": "stub"}, "endpoint": {"base_url": url}}},
        "roles": {"answer": "llm"},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _run(suites: Path, profile: Path, out: Path) -> int:
    return main(["run", "--suite", "xr_render_scenes", "--evals-dir", str(suites), "--profile", str(profile),
                 "--out", str(out), "--no-gpu-memory"])  # fmt: skip


def test_suite_is_the_whole_original_corpus_and_the_file_is_fresh():
    corpus = _load("drivers").harness().CORPUS_CASES
    suite = load_suite(SUITES, "xr_render_scenes")
    assert [c.id for c in suite.cases] == [c["name"] for c in corpus] and len(corpus) == 69
    assert {c.role for c in suite.cases} == {"answer"} and all(c.skip_reason is None for c in suite.cases)
    generated = _load("generate_cases").render()
    assert (SUITE / "cases" / "scenarios.yaml").read_text(encoding="utf-8") == generated


def test_scorer_applies_the_original_checker():
    scorers, drivers = _load("scorers"), _load("drivers")
    case = type("Case", (), {"params": {"case": "make_red_sphere"}})()
    sphere = ("add_primitive", {"prim_type": "sphere", "r": 1.0, "g": 0.0, "b": 0.0, "x": 0, "y": 1.6, "z": -1.5})
    good = type("Result", (), {"text": json.dumps({"calls": [sphere], "reply": "Done."})})()
    bad = type("Result", (), {"text": json.dumps({"calls": [], "reply": "Done."})})()
    assert scorers.scenario(case, good)[0] == 1.0
    score, evidence = scorers.scenario(case, bad)
    assert score == 0.0 and "unmatched: add_primitive" in evidence["why"]
    assert drivers.corpus_case("make_red_sphere")["user"] == "Make a red sphere."
    with pytest.raises(ValueError):
        drivers.corpus_case("nope")


def test_scenario_runs_the_real_supervisor_against_the_stub_and_passes(llm, tmp_path):
    server = llm(
        [
            _call("object_agent", instruction="Create a red sphere in front of the user."),
            _call("create_at", prim_type="sphere", x=0.0, y=1.6, z=-1.5, color_kind="literal", color_value="red"),
            {"content": "Created a red sphere."},
            {"content": "Done, a red sphere is in front of you."},
        ]
    )
    suites = _one_case_suite(tmp_path, "make_red_sphere")
    out = tmp_path / "r.jsonl"
    assert _run(suites, _profile(tmp_path, server.url), out) == 0
    (record,) = read_records(out)
    identity = (record.suite, record.case, record.role, record.passed)
    assert identity == ("xr_render_scenes", "make_red_sphere", "answer", True)
    assert "add_primitive" in record.detail["tools"]
    assert any(r.get("tools") for r in server.requests)  # the supervisor sent tool definitions to the model


def test_scenario_that_changes_nothing_fails_without_erroring(llm, tmp_path):
    server = llm([{"content": "I made a red sphere."}] * 3)
    suites = _one_case_suite(tmp_path, "make_red_sphere")
    out = tmp_path / "r.jsonl"
    assert _run(suites, _profile(tmp_path, server.url), out) == 1
    (record,) = read_records(out)
    assert not record.passed and record.error is None
