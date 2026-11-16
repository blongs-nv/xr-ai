# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scorer types: exact, contains, regex, choice, json_match, judge and the python escape hatch."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from xr_ai_eval.hooks import call_hook, load_hook, split_spec

SCORER_TYPES = ("exact", "contains", "regex", "choice", "json_match", "judge", "python")

# A judge call: (role, prompt) -> (reply text, preset that answered).
JudgeFn = Callable[[str, str], Awaitable[tuple[str, str]]]


def validate_scorer(scorer: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a scorer block is incomplete or inconsistent."""
    kind = scorer["type"]
    if kind == "python":
        split_spec(scorer.get("function"))
        return
    if kind == "judge":
        if not isinstance(scorer.get("role"), str) or not isinstance(scorer.get("rubric"), str):
            raise ValueError("judge scorer needs 'role' and 'rubric'")
        return
    if "expect" not in scorer:
        raise ValueError(f"{kind} scorer needs 'expect'")
    expect = scorer["expect"]
    if kind == "regex":
        if not isinstance(expect, str):
            raise ValueError("regex 'expect' must be a string")
        try:
            re.compile(expect)
        except re.error as exc:
            raise ValueError(f"regex 'expect' does not compile: {exc}") from exc
    elif kind == "choice":
        choices = scorer.get("choices")
        if not isinstance(choices, list) or not all(isinstance(item, str) for item in choices):
            raise ValueError("choice scorer needs a 'choices' list of strings")
        if expect not in choices:
            raise ValueError(f"choice 'expect' {expect!r} is not one of {choices}")
    elif kind == "json_match":
        if not isinstance(expect, dict) or not expect:
            raise ValueError("json_match 'expect' must be a non-empty mapping")
    elif kind == "contains":
        if not isinstance(expect, str | list) or not expect:
            raise ValueError("contains 'expect' must be a string or a non-empty list of strings")
    elif not isinstance(expect, str):
        raise ValueError("exact 'expect' must be a string")


async def score_output(
    scorer: dict[str, Any],
    output: str,
    judge: JudgeFn | None = None,
    *,
    case: Any = None,
    result: Any = None,
) -> tuple[float, dict[str, Any]]:
    """Score one model output. Returns ``(score, evidence)`` with ``score`` in 0.0 to 1.0.

    ``case`` and ``result`` are passed to ``python`` scorers only.
    """
    kind = scorer["type"]
    detail: dict[str, Any] = {"output": output}
    if kind == "python":
        return await _python(scorer, case, result, detail)
    if kind == "exact":
        return float(output.strip() == scorer["expect"].strip()), detail
    if kind == "contains":
        needles = [scorer["expect"]] if isinstance(scorer["expect"], str) else list(scorer["expect"])
        missing = [needle for needle in needles if needle not in output]
        if missing:
            detail["missing"] = missing
        return float(not missing), detail
    if kind == "regex":
        return float(re.search(scorer["expect"], output) is not None), detail
    if kind == "choice":
        picked = pick_choice(output, scorer["choices"])
        detail["picked"] = picked
        return float(picked == scorer["expect"]), detail
    if kind == "json_match":
        return _json_match(scorer["expect"], output, detail)
    return await _judge(scorer, output, judge, detail)


def pick_choice(output: str, choices: list[str]) -> str | None:
    """Return the one choice the output names, or ``None`` when none or several differ."""
    text = output.strip().strip("\"'`.!").lower()
    for choice in choices:
        if text == choice.lower():
            return choice
    found = [c for c in choices if re.search(rf"(?<![\w-]){re.escape(c.lower())}(?![\w-])", output.lower())]
    return found[0] if len(found) == 1 else None


def _extract_json(output: str) -> Any:
    text = output.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _json_match(expect: dict[str, Any], output: str, detail: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    parsed = _extract_json(output)
    if not isinstance(parsed, dict):
        detail["parse_error"] = "output is not a JSON object"
        return 0.0, detail
    wrong = sorted(key for key, value in expect.items() if parsed.get(key) != value)
    if wrong:
        detail["wrong_fields"] = wrong
    return (len(expect) - len(wrong)) / len(expect), detail


async def _judge(
    scorer: dict[str, Any], output: str, judge: JudgeFn | None, detail: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    if judge is None:
        raise RuntimeError("judge scorer needs a judge function")
    prompt = (
        "Score the answer against the rubric. Reply with only a number from 0 to 1.\n\n"
        f"Rubric: {scorer['rubric']}\n\nAnswer:\n{output}"
    )
    reply, preset = await judge(scorer["role"], prompt)
    detail["judge_role"] = scorer["role"]
    detail["judge_preset"] = preset
    number = re.search(r"\d*\.?\d+", reply)
    if number is None:
        detail["judge_reply"] = reply
        raise RuntimeError(f"judge reply has no number: {reply!r}")
    return min(1.0, max(0.0, float(number.group()))), detail


async def _python(
    scorer: dict[str, Any], case: Any, result: Any, detail: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Run ``function(case, result) -> score | (score, evidence)``; the file is relative to the case file."""
    function = load_hook(case.base_dir, scorer["function"])
    value = await call_hook(function, case, result)
    evidence: dict[str, Any] = {}
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        value, evidence = value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"python scorer must return a number from 0 to 1, got {value!r}")
    detail.update(evidence)
    return min(1.0, max(0.0, float(value))), detail
