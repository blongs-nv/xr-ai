# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The six existing harnesses: how to run each and how to read what it reports.

Parsers read the exact formats the harnesses print or write today. A harness that changes its output
format breaks its adapter test, not silently the scorecard.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from xr_ai_eval_adapters.base import (
    LATENCY_GAP,
    LATENCY_REPORTED,
    HarnessAdapter,
    HarnessRun,
    Parsed,
    Step,
    gaps_ms,
)

_TEA = "agent-samples/tea-making-sample"
_LAB = "agent-samples/lab-instrument-monitoring"
_CONV = "agent-samples/conversation-example"
_RENDER = "agent-samples/xr-render-demo"
_PCG = "agent-samples/procedure-capture-guide"
_CONTEXT = "agent-sdk/xr-ai-conversation"


def _names(path: str, key: str | None = None) -> list[str]:
    """Case names from a harness's own YAML, so cases that never reported become error records."""

    def read(root: Path) -> list[str]:
        try:
            doc = yaml.safe_load((root / path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return []
        rows = doc.get(key) if key and isinstance(doc, dict) else doc
        return [str(r["name"]) for r in rows or [] if isinstance(r, dict) and "name" in r]

    return read


def _timed(run: HarnessRun, regex: re.Pattern[str]) -> list[tuple[re.Match[str], float]]:
    """Result lines matching ``regex`` with the ms since the previous result line."""
    hits = [(m, line.at_s) for line in run.lines if (m := regex.match(line.text))]
    return [(m, ms) for (m, _), ms in zip(hits, gaps_ms([at for _, at in hits]), strict=True)]


# ---- tea-making-sample: "PASS|MISS name: tools=[...] content='...'" -------------------------------
_TEA_LINE = re.compile(r"^(PASS|MISS) (\S+): tools=(.*?) content=(.*)$")


def parse_tea(run: HarnessRun, work: Path) -> list[Parsed]:
    """Read ``agent-samples/tea-making-sample/eval/eval.py`` output."""
    return [
        Parsed(
            m[2],
            m[1] == "PASS",
            1.0 if m[1] == "PASS" else 0.0,
            ms,
            LATENCY_GAP,
            detail={"tools": m[3], "content": m[4]},
        )
        for m, ms in _timed(run, _TEA_LINE)
    ]


# ---- lab-instrument-monitoring: eval.py and visual_eval.py ------------------------------------------
_LAB_TOOLS = re.compile(r"^(PASS|FAIL) (\S+): tools=(.*)$")
_LAB_VISUAL = re.compile(r"^(PASS|FAIL) (\S+): (.*)$")


def parse_lab_tools(run: HarnessRun, work: Path) -> list[Parsed]:
    """Read ``eval/eval.py`` output; a failing case is followed by an indented ``content=`` line."""
    out = []
    for m, ms in _timed(run, _LAB_TOOLS):
        out.append(Parsed(m[2], m[1] == "PASS", float(m[1] == "PASS"), ms, LATENCY_GAP, detail={"tools": m[3]}))
    return out


def parse_lab_visual(run: HarnessRun, work: Path) -> list[Parsed]:
    """Read ``eval/visual_eval.py`` output (``PASS|FAIL name: 'response'``)."""
    return [
        Parsed(m[2], m[1] == "PASS", float(m[1] == "PASS"), ms, LATENCY_GAP, detail={"response": m[3]})
        for m, ms in _timed(run, _LAB_VISUAL)
    ]


# ---- conversation-example: --output JSONL, one record per turn --------------------------------------
_CONV_LINE = re.compile(r"^(PASS|MISS) (\S+)(?: (.*))?$")


def parse_conversation_example(run: HarnessRun, work: Path) -> list[Parsed]:
    """Read the ``--output`` JSONL; fall back to the printed lines when the file was not written."""
    path = work / "records.jsonl"
    if path.is_file():
        timed = {m[2]: ms for m, ms in _timed(run, _CONV_LINE)}
        out = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            checks = row.get("checks") or {}
            score = sum(bool(v) for v in checks.values()) / len(checks) if checks else float(row["passed"])
            ms = timed.get(row["turn"])
            out.append(
                Parsed(
                    row["turn"],
                    bool(row["passed"]),
                    1.0 if row["passed"] else min(score, 0.99),
                    ms,
                    LATENCY_GAP if ms is not None else None,
                    error=row.get("error"),
                    detail={"checks": checks, "answered": row.get("answered"), "decisions": row.get("decisions")},
                )
            )
        return out
    return [
        Parsed(m[2], m[1] == "PASS", float(m[1] == "PASS"), ms, LATENCY_GAP, detail={"note": m[3] or ""})
        for m, ms in _timed(run, _CONV_LINE)
    ]


# ---- xr-ai-conversation: conflict_eval.py and compaction_eval.py -----------------------------------
_CONFLICT = re.compile(r"^(PASS|FAIL) (\S+): (\d+)/(\d+) \| (.*)$")
_SEEDING = re.compile(r"^FAIL (\S+): seeding (.*)$")
_SESSION = re.compile(r"^=== (\S+)$")
_PROBE = re.compile(r"^(PASS|FAIL) (\S+): in prompt=(True|False) answers (\d+)/(\d+) \| (.*)$")


def parse_conflicts(run: HarnessRun, work: Path) -> list[Parsed]:
    """Read ``eval/conflict_eval.py``: score is the share of sampled answers that were correct."""
    out = []
    previous = 0.0
    for line in run.lines:
        if m := _CONFLICT.match(line.text):
            ms = (line.at_s - previous) * 1000.0
            previous = line.at_s
            out.append(
                Parsed(
                    m[2],
                    m[1] == "PASS",
                    int(m[3]) / int(m[4]),
                    ms,
                    LATENCY_GAP,
                    detail={"correct": int(m[3]), "samples": int(m[4]), "first_answer": m[5]},
                )
            )
        elif m := _SEEDING.match(line.text):
            previous = line.at_s
            out.append(Parsed(m[1], False, 0.0, error=f"seeding failed: {m[2]}"))
    return out


def parse_compaction(run: HarnessRun, work: Path) -> list[Parsed]:
    """Read ``eval/compaction_eval.py``: probes are scored per session, ``session/probe``."""
    out = []
    session, previous = "session", 0.0
    for line in run.lines:
        if m := _SESSION.match(line.text):
            session = m[1]
        elif m := _PROBE.match(line.text):
            ms = (line.at_s - previous) * 1000.0
            out.append(
                Parsed(
                    f"{session}/{m[2]}",
                    m[1] == "PASS",
                    int(m[4]) / int(m[5]),
                    ms,
                    LATENCY_GAP,
                    detail={"in_prompt": m[3] == "True", "correct": int(m[4]), "samples": int(m[5])},
                )
            )
        if _PROBE.match(line.text) or line.text.startswith("turn "):
            previous = line.at_s
    return out


# ---- xr-render-demo: "{status:32} name: response" ---------------------------------------------------
_RENDER_LINE = re.compile(r"^(PASS|FAIL\b.*?)\s+([a-z0-9][\w\-]*): (.*)$")


def parse_render(run: HarnessRun, work: Path) -> list[Parsed]:
    """Read ``xr_render_demo_eval`` output. A FAIL status carries its reason and can exceed 32 columns."""
    out = []
    for m, ms in _timed(run, _RENDER_LINE):
        ok = m[1] == "PASS"
        detail = {"response": m[3]} if ok else {"failure": m[1].removeprefix("FAIL").strip(), "response": m[3]}
        out.append(Parsed(m[2], ok, float(ok), ms, LATENCY_GAP, detail=detail))
    return out


# ---- procedure-capture-guide: visual checks (results.jsonl) and caption contract (one JSON per case) -
def parse_visual_checks(run: HarnessRun, work: Path) -> list[Parsed]:
    """Read ``results.jsonl`` from ``evaluate_visual_checks.py``; ``latency_s`` is reported per row."""
    path = work / "out" / "results.jsonl"
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        case = f"{row['scene']}/{row['step']}/trial{row['trial']}"
        detail = {"answer": row.get("answer"), "expected_answer": row.get("expected_answer")}
        latency = row.get("latency_s")
        ms = None if latency is None else latency * 1000.0
        if row.get("status") != "scored":
            out.append(Parsed(case, False, 0.0, ms, LATENCY_REPORTED, error="VLM unavailable", detail=detail))
            continue
        parts = [row.get("recognition") is True, row.get("gate_correct") is True]
        detail.update(recognition=row.get("recognition"), gate_correct=row.get("gate_correct"))
        out.append(Parsed(case, all(parts), sum(parts) / 2, ms, LATENCY_REPORTED, detail=detail))
    return out


def parse_captions(run: HarnessRun, work: Path) -> list[Parsed]:
    """Read the per-case JSON files from ``evaluate_captions.py`` (structural validity only)."""
    out = []
    for path in sorted((work / "out").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or "case" not in row:
            continue
        ok = "error" not in row and row.get("available", False)
        detail = {"expected": row.get("expected"), "raw_response": row.get("raw_response")}
        out.append(Parsed(str(row["case"]), ok, float(ok), error=row.get("error"), detail=detail))
    return out


def _out_arg(ctx: object, work: Path) -> list[str]:
    return ["--output", str(work / "out")]


def _records_arg(ctx: object, work: Path) -> list[str]:
    return ["--output", str(work / "records.jsonl")]


def _visual_args(ctx: object, work: Path) -> list[str]:
    manifest = f"{_PCG}/example-guides/tea-making/evaluation/fixtures/visual_expectations.yaml"
    return ["--manifest", manifest, "--output", str(work / "out"), "--repeats", "1"]


TEA_MAKING = HarnessAdapter(
    "tea-making",
    [
        Step(
            "", "answer", f"{_TEA}/eval/eval.py", f"{_TEA}/worker", parse_tea,
            sample_models=f"{_TEA}/yaml/models.json",
            expected=_names(f"{_TEA}/eval/cases.yaml"),
        )
    ],
)
LAB_INSTRUMENT = HarnessAdapter(
    "lab-instrument-monitoring",
    [
        Step(
            "tools", "answer", f"{_LAB}/eval/eval.py", f"{_LAB}/worker", parse_lab_tools,
            sample_models=f"{_LAB}/yaml/models.json",
            expected=_names(f"{_LAB}/eval/cases.yaml"),
        ),
        Step(
            "visual", "perceive", f"{_LAB}/eval/visual_eval.py", f"{_LAB}/worker", parse_lab_visual,
            sample_models=f"{_LAB}/yaml/models.json",
            expected=_names(f"{_LAB}/eval/visual_cases.yaml"),
        ),
    ],
)
CONVERSATION_EXAMPLE = HarnessAdapter(
    "conversation-example",
    [
        Step(
            "", "decide", f"{_CONV}/eval/run_decision_eval.py", f"{_CONV}/worker", parse_conversation_example,
            sample_models=f"{_CONV}/yaml/models.json",
            extra_args=_records_arg,
        )
    ],
)
CONVERSATION_CONTEXT = HarnessAdapter(
    "conversation-context",
    [
        Step(
            "conflict", "answer", f"{_CONTEXT}/eval/conflict_eval.py", _CONTEXT, parse_conflicts,
            uv_args=("--group", "eval"), models_flag="--models", model_flag="--model",
            expected=_names(f"{_CONTEXT}/eval/cases/conflicts.yaml", "cases"),
        ),
        Step(
            "compaction", "answer", f"{_CONTEXT}/eval/compaction_eval.py", _CONTEXT, parse_compaction,
            uv_args=("--group", "eval"), models_flag="--models", model_flag="--model",
        ),
    ],
)
XR_RENDER_DEMO = HarnessAdapter(
    "xr-render-demo",
    [Step("", "answer", "-m xr_render_demo_eval.harness", f"{_RENDER}/eval", parse_render)],
)
PROCEDURE_CAPTURE_GUIDE = HarnessAdapter(
    "procedure-capture-guide",
    [
        Step(
            "visual", "perceive", f"{_PCG}/example-guides/tea-making/evaluation/evaluate_visual_checks.py",
            f"{_PCG}/worker", parse_visual_checks, models_flag="--models", extra_args=_visual_args,
        ),
        Step(
            "caption", "perceive", f"{_PCG}/evaluation/capture/evaluate_captions.py", f"{_PCG}/worker",
            parse_captions, extra_args=_out_arg,
        ),
    ],
)

ADAPTERS = {
    a.name: a
    for a in (
        TEA_MAKING,
        LAB_INSTRUMENT,
        CONVERSATION_EXAMPLE,
        CONVERSATION_CONTEXT,
        XR_RENDER_DEMO,
        PROCEDURE_CAPTURE_GUIDE,
    )
}
