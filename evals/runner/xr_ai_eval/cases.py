# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate ``evals/<suite>/cases/*.yaml`` (``evals/EVAL_SCHEMA.md``)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xr_ai_eval.hooks import resolve_hook_path, split_spec
from xr_ai_eval.scorers import SCORER_TYPES, validate_scorer


class CaseError(ValueError):
    """A case file is invalid. The runner exits with code 2."""


@dataclass(frozen=True)
class Turn:
    """One user turn of a multi-turn case."""

    text: str
    frames: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Case:
    """One validated case with fixture paths resolved."""

    id: str
    role: str
    text: str
    scorer: dict[str, Any]
    frames: tuple[Path, ...] = ()
    context: tuple[dict[str, str], ...] = ()
    pass_threshold: float = 1.0
    tags: tuple[str, ...] = ()
    skip_reason: str | None = None
    system: str | None = None
    turns: tuple[Turn, ...] = ()
    audio: Path | None = None
    driver: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    base_dir: Path = Path()


@dataclass(frozen=True)
class Suite:
    """All cases of one suite."""

    name: str
    version: int
    description: str
    cases: tuple[Case, ...] = field(default_factory=tuple)


def load_suite(evals_dir: Path, name: str) -> Suite:
    """Load every ``cases/*.yaml`` of suite ``name`` under ``evals_dir``."""
    files = sorted((evals_dir / name / "cases").glob("*.yaml"))
    if not files:
        raise CaseError(f"no case files in {evals_dir / name / 'cases'}")
    cases: list[Case] = []
    seen: dict[str, Path] = {}
    version = 1
    description = ""
    for path in files:
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise CaseError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(doc, dict):
            raise CaseError(f"{path}: expected a mapping at the top level")
        if doc.get("suite") != name:
            raise CaseError(f"{path}: suite is {doc.get('suite')!r}, expected {name!r} (the directory name)")
        if not isinstance(doc.get("version"), int):
            raise CaseError(f"{path}: 'version' is required and must be an integer")
        version = max(version, doc["version"])
        description = description or str(doc.get("description") or "")
        defaults = doc.get("defaults") or {}
        if not isinstance(defaults, dict):
            raise CaseError(f"{path}: 'defaults' must be a mapping")
        raw_cases = doc.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise CaseError(f"{path}: 'cases' must be a non-empty list")
        for raw in raw_cases:
            case = _build_case(path, defaults, raw)
            if case.id in seen:
                raise CaseError(f"{path}: duplicate case id {case.id!r} (also in {seen[case.id]})")
            seen[case.id] = path
            cases.append(case)
    return Suite(name=name, version=version, description=description, cases=tuple(cases))


def _build_case(path: Path, defaults: dict[str, Any], raw: Any) -> Case:
    if not isinstance(raw, dict):
        raise CaseError(f"{path}: every case must be a mapping")
    merged = {**defaults, **raw}
    case_id = merged.get("id")
    if not isinstance(case_id, str) or not re.fullmatch(r"[\w.\-]+", case_id):
        raise CaseError(f"{path}: case 'id' is required and must be a simple identifier, got {case_id!r}")
    where = f"{path}: case {case_id!r}"
    role = merged.get("role")
    if not isinstance(role, str) or not role:
        raise CaseError(f"{where}: 'role' is required (set it on the case or in defaults)")
    source = merged.get("input")
    if not isinstance(source, dict):
        raise CaseError(f"{where}: 'input' is required")
    skip = merged.get("skip_reason")
    skipped = skip is not None  # a skipped case may name media that is not recorded yet
    has_turns, audio_name, driver = source.get("turns"), source.get("audio"), source.get("driver")
    if not isinstance(source.get("text"), str) and not (has_turns or audio_name or driver):
        raise CaseError(f"{where}: 'input.text' is required (or 'input.turns', 'input.audio' or 'input.driver')")
    if has_turns and (source.get("text") is not None or source.get("frames")):
        raise CaseError(f"{where}: with 'input.turns', 'text' and 'frames' belong to each turn")
    scorer = merged.get("scorer")
    if not isinstance(scorer, dict) or scorer.get("type") not in SCORER_TYPES:
        raise CaseError(f"{where}: scorer.type must be one of {', '.join(SCORER_TYPES)}")
    try:
        validate_scorer(scorer)
    except ValueError as exc:
        raise CaseError(f"{where}: {exc}") from exc
    if scorer["type"] == "python":
        _require_hook(where, path, scorer["function"], skipped)
    if driver is not None:
        try:
            split_spec(driver)
        except ValueError as exc:
            raise CaseError(f"{where}: input.driver: {exc}") from exc
        _require_hook(where, path, driver, skipped)
    params = source.get("params") or {}
    if not isinstance(params, dict):
        raise CaseError(f"{where}: input.params must be a mapping")
    system = merged.get("system")
    if system is not None and not isinstance(system, str):
        raise CaseError(f"{where}: 'system' must be a string")
    threshold = merged.get("pass_threshold", 1.0)
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0.0 <= threshold <= 1.0:
        raise CaseError(f"{where}: pass_threshold must be a number from 0.0 to 1.0")
    frames = _frames(where, path, source.get("frames"), skipped)
    turns = []
    if has_turns:
        if not isinstance(has_turns, list):
            raise CaseError(f"{where}: input.turns must be a list")
        for turn in has_turns:
            if not isinstance(turn, dict) or not isinstance(turn.get("text"), str):
                raise CaseError(f"{where}: each turn needs 'text'")
            turns.append(Turn(turn["text"], _frames(where, path, turn.get("frames"), skipped)))
    audio = None
    if audio_name is not None:
        audio = (path.parent / str(audio_name)).resolve()
        if not audio.is_file() and not skipped:
            raise CaseError(f"{where}: audio {audio_name!r} not found at {audio}")
    context = []
    for message in source.get("context") or []:
        if not isinstance(message, dict) or not isinstance(message.get("text"), str):
            raise CaseError(f"{where}: each context message needs 'role' and 'text'")
        context.append({"role": str(message.get("role", "user")), "text": message["text"]})
    tags = merged.get("tags") or []
    if not isinstance(tags, list):
        raise CaseError(f"{where}: tags must be a list")
    return Case(
        id=case_id,
        role=role,
        text=str(source.get("text") or ""),
        scorer=scorer,
        frames=frames,
        context=tuple(context),
        pass_threshold=float(threshold),
        tags=tuple(str(tag) for tag in tags),
        skip_reason=None if skip is None else str(skip),
        system=system,
        turns=tuple(turns),
        audio=audio,
        driver=driver,
        params=params,
        base_dir=path.parent,
    )


def _frames(where: str, path: Path, names: Any, skipped: bool) -> tuple[Path, ...]:
    if names is not None and not isinstance(names, list):
        raise CaseError(f"{where}: frames must be a list")
    frames = []
    for frame in names or []:
        frame_path = (path.parent / str(frame)).resolve()
        if not frame_path.is_file() and not skipped:
            raise CaseError(f"{where}: frame {frame!r} not found at {frame_path}")
        frames.append(frame_path)
    return tuple(frames)


def _require_hook(where: str, path: Path, spec: str, skipped: bool) -> None:
    try:
        hook = resolve_hook_path(path.parent, spec)
    except ValueError as exc:
        raise CaseError(f"{where}: {exc}") from exc
    if not hook.is_file() and not skipped:
        raise CaseError(f"{where}: hook file not found at {hook}")
