# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech schedules reuse a sample scenario's inputs, media and assertions."""

from __future__ import annotations

import dataclasses
import hashlib
import math

import yaml

from .scenario import Scenario, Step


def _utterance(text, *, reply=True, handler=True):
    return {
        "path": hashlib.sha256(text.encode()).hexdigest() + ".wav",
        "text": text,
        "handler": handler,
        "reply": reply,
    }


def _load_speech(raw, path):
    if set(raw) - {"scenario", "duration_s", "schedule", "controls", "no_reply", "unprompted"}:
        raise ValueError("speech replay accepts scenario, duration_s, schedule, controls, no_reply, unprompted")
    if not isinstance(raw["scenario"], str) or not raw["scenario"]:
        raise ValueError("speech scenario must name a source file")
    source_path = (path.parent / raw["scenario"]).resolve()
    source_raw = yaml.safe_load(source_path.read_text())
    source = Scenario.from_dict(source_raw, base_dir=source_path.parent, origin=str(source_path))
    schedule = raw.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != len(source.steps):
        raise ValueError("speech schedule needs one time for every source scenario step")
    if any(type(at) not in (int, float) or not math.isfinite(at) or at < 0 for at in schedule) or schedule != sorted(
        schedule
    ):
        raise ValueError("speech schedule times must be nonnegative and ordered")
    no_reply = raw.get("no_reply", [])
    if not isinstance(no_reply, list) or any(
        type(i) is not int or not 0 <= i < len(source.steps) or source.steps[i].action != "say" for i in no_reply
    ):
        raise ValueError("no_reply must identify source say steps")
    unprompted = raw.get("unprompted", [])
    if not isinstance(unprompted, list) or any(
        type(i) is not int or not 0 <= i < len(source.steps) or source.steps[i].action not in {"expect", "wait_for"}
        for i in unprompted
    ):
        raise ValueError("unprompted must identify source check steps")
    steps = []
    for index, (step, at) in enumerate(zip(source.steps, schedule, strict=True)):
        value, action = step.value, step.action
        if action == "say":
            action, value = "audio", _utterance(str(value), reply=index not in no_reply)
        elif index in unprompted:
            value = {**(value if isinstance(value, dict) else {"said": value}), "unprompted": True}
        elif action == "frames":
            spec = value if isinstance(value, dict) else {"dir": value}
            fps = spec.get("fps", 1)
            if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("frame replay fps must be positive")
            files = sorted(
                p
                for p in (source.base_dir / spec["dir"]).iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
            if not files:
                raise ValueError("frame replay directory contains no images")
            for offset, file in enumerate(files):
                steps.append(
                    Step(
                        "image",
                        str(file.relative_to(source.base_dir)),
                        at=at + offset / fps,
                        participant=step.participant,
                    )
                )
            continue
        steps.append(dataclasses.replace(step, action=action, value=value, at=at, relative=False))
    controls = raw.get("controls", [])
    if not isinstance(controls, list):
        raise ValueError("speech controls must be a list")
    for control in controls:
        if (
            not isinstance(control, dict)
            or set(control) != {"at", "text"}
            or not isinstance(control["text"], str)
            or not control["text"].strip()
        ):
            raise ValueError("speech controls require at and nonempty text")
        if type(control["at"]) not in (int, float) or not math.isfinite(control["at"]) or control["at"] < 0:
            raise ValueError("speech control time must be finite and nonnegative")
        steps.append(Step("audio", _utterance(control["text"], reply=False, handler=False), at=control["at"]))
    steps.sort(key=lambda step: step.at)
    return dataclasses.replace(
        source, name="Speech: " + source.name, steps=tuple(steps), duration_s=raw.get("duration_s")
    )
