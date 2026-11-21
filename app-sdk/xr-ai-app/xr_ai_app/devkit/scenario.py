# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenarios: what a user does, with timestamps, as a file.

.. code-block:: yaml

    name: kettle happy path
    participant: p1          # default participant for steps that do not name one
    steps:
      - join: p1
      - at: 0.2
        image: frames/tea_box.jpg
      - image: {path: frames/label.jpg, stream: false}   # held, handed over only when the app asks
      - at: 1.0
        say: start tea making
      - frames: {dir: frames/kettle/, fps: 1}
      - video: {path: clips/kettle.mp4, fps: 2, from_s: 0, to_s: 12, background: true}
      - wait_for: {said: "kettle has water", within_s: 20}
      - at: +1.0             # relative to the previous step
        say: what step am I on?
      - expect: {said: heat, within_s: 10}
      - leave: p1

Each step has exactly one action (``join``, ``leave``, ``say``, ``audio``, ``image``, ``frames``, ``video``,
``wait_for``, ``expect``, ``sleep``) plus optional ``at`` (seconds from the start, or ``+s`` after the
previous step) and ``participant``. ``image`` streams one frame; ``image: {path: ..., stream: false}`` streams
nothing and holds the picture for the app to ask for (a client in on-demand camera mode). ``wait_for`` and
``expect`` are the same check (some output after the previous user action contains ``said``, case-insensitive,
within ``within_s``; ``said_any: [a, b]`` accepts any of several, for real models that word things
differently); ``expect`` reads as an assertion. Media paths resolve against the scenario file, or first
against ``--media-dir`` (your own recordings under the same names). ``audio`` is a WAV path mapping for
microphone replay only; ``duration_s`` optionally fixes that timeline.
``xr-ai-dev prepare`` generates and pins media before ``run --real-audio --real-models --clock real``.
A ``video`` or ``frames`` step plays to its
end before the next step, or with ``background: true`` keeps playing while the next steps run (talking over a
recording); background media stops when its participant leaves or the scenario ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ACTIONS", "Scenario", "Step"]

ACTIONS = ("join", "leave", "say", "audio", "image", "frames", "video", "wait_for", "expect", "sleep")


@dataclass(frozen=True)
class Step:
    action: str
    value: Any
    at: float | None = None
    """Seconds from the scenario start; ``None``: right after the previous step."""
    relative: bool = False
    """``at`` counts from the previous step instead of the start."""
    participant: str | None = None

    def describe(self) -> str:
        return f"{self.action}: {self.value}"


@dataclass(frozen=True)
class Scenario:
    name: str
    steps: tuple[Step, ...]
    participant: str = "p1"
    base_dir: Path = field(default=Path("."))

    duration_s: float | None = None

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        file = Path(path).expanduser().resolve()
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict) and "scenario" in raw:
            from ._speech_scenario import _load_speech  # noqa: PLC0415

            return _load_speech(raw, file)
        return cls.from_dict(raw, base_dir=file.parent, origin=str(file))

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base_dir: Path = Path("."), origin: str = "scenario") -> Scenario:
        if not isinstance(raw, dict):
            raise ValueError(f"{origin}: a scenario is a mapping")
        if unknown := set(raw) - {"name", "participant", "steps", "duration_s"}:
            raise ValueError(f"{origin}: unknown keys {sorted(unknown)}")
        steps = []
        for index, item in enumerate(raw.get("steps") or []):
            if not isinstance(item, dict):
                raise ValueError(f"{origin}: step {index} is not a mapping")
            actions = [k for k in item if k in ACTIONS]
            extra = set(item) - set(ACTIONS) - {"at", "participant"}
            if len(actions) != 1 or extra:
                raise ValueError(f"{origin}: step {index} needs exactly one of {ACTIONS} (got {sorted(item)})")
            at, relative = _at(item.get("at"), origin, index)
            steps.append(Step(actions[0], item[actions[0]], at, relative, item.get("participant")))
        return cls(
            duration_s=raw.get("duration_s"),
            name=str(raw.get("name") or origin),
            steps=tuple(steps),
            participant=str(raw.get("participant") or "p1"),
            base_dir=base_dir,
        )


def _at(value: Any, origin: str, index: int) -> tuple[float | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, str) and value.strip().startswith("+"):
        return float(value.strip()[1:]), True
    try:
        return float(value), False
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{origin}: step {index}: `at` must be seconds or +seconds, got {value!r}") from exc
