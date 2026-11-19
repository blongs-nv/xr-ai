# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The evidence packet of one capture: pure state and rendering, no I/O and no tasks.

The packet format is that of ``agent-samples/procedure-capture-guide`` (``packet.json``, ``summary.md``,
``frames/index.jsonl``, ``captions.jsonl``, ``transcript.jsonl``, ``errors.jsonl``), unchanged, because its
``guide-generation-skills`` and ``evaluation/guide`` read it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from xr_ai_app import Frame

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def iso(timestamp_us: int) -> str:
    return datetime.fromtimestamp(timestamp_us / 1_000_000, UTC).isoformat()


def path_safe(value: str) -> str:
    """``value`` with anything but letters, digits, ``_``, ``.`` and ``-`` replaced, for a file or folder name."""
    return _SAFE.sub("-", value).strip("-._") or "participant"


def session_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def elapsed_hms(timestamp_us: int, started_at_us: int) -> str:
    """The time from ``started_at_us`` to ``timestamp_us`` as ``hh:mm:ss``."""
    elapsed = max(0, timestamp_us - started_at_us) // 1_000_000
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


@dataclass(slots=True)
class SavedFrame:
    frame_id: int
    timestamp_us: int
    path: str
    sequence: int
    width: int
    height: int
    frame: Frame | None = None
    """The live frame, kept in memory for captioning (so the saved file is not read back)."""


@dataclass(slots=True)
class Phase:
    name: str
    started_at_us: int
    ended_at_us: int
    caption_count: int
    summary: str
    latest_delta: str
    evidence_frame_id: int
    evidence_path: str


@dataclass(slots=True)
class Activity:
    name: str
    started_at_us: int
    ended_at_us: int
    caption_count: int
    phases: list[Phase] = field(default_factory=list)


@dataclass(slots=True)
class CaptureSession:
    """Everything one demonstration accumulates. Lives in ``p.state``; mutated before any ``await``."""

    participant_id: str
    session_id: str
    started_at: str
    started_at_us: int
    frame_count: int = 0
    transcript_count: int = 0
    narration_status: Literal["pending", "complete", "failed"] = "pending"
    caption_count: int = 0
    last_frame_key: tuple[int, int] | None = None
    last_captioned_frame: SavedFrame | None = None
    latest_frame: SavedFrame | None = None
    previous_caption: dict[str, str] | None = None
    activities: list[Activity] = field(default_factory=list)
    # (timestamp_us, "Heard" | "Saw", text): narration and captions merged by time.
    timeline: list[tuple[int, str, str]] = field(default_factory=list)
    active: bool = True
    status: Literal["recording", "complete", "incomplete"] = "recording"
    ended_at: str | None = None
    stop_command: dict[str, Any] | None = None
    sealed: bool = False


def update_hierarchy(state: CaptureSession, caption: dict[str, Any]) -> None:
    """Group contiguous matching labels. A phase's summary keeps its latest caption."""
    timestamp_us = int(caption["frame_timestamp_us"])
    activity_name, phase_name = str(caption["activity"]), str(caption["phase"])
    if not state.activities or state.activities[-1].name != activity_name:
        state.activities.append(Activity(activity_name, timestamp_us, timestamp_us, 0))
    state.timeline.append((timestamp_us, "Saw", f"{caption['phase']}: {caption['caption']}"))
    activity = state.activities[-1]
    activity.ended_at_us = timestamp_us
    activity.caption_count += 1
    if not activity.phases or activity.phases[-1].name != phase_name:
        activity.phases.append(
            Phase(
                phase_name,
                timestamp_us,
                timestamp_us,
                0,
                "",
                "",
                int(caption["frame_id"]),
                str(caption["frame_path"]),
            )
        )
    phase = activity.phases[-1]
    phase.ended_at_us = timestamp_us
    phase.caption_count += 1
    phase.summary = str(caption["caption"])
    phase.latest_delta = str(caption["delta"])
    phase.evidence_frame_id = int(caption["frame_id"])
    phase.evidence_path = str(caption["frame_path"])


def packet(state: CaptureSession, *, capture_fps: float, caption_interval_s: float) -> dict[str, Any]:
    """The machine-readable index written as ``packet.json``."""
    return {
        "schema_version": 1,
        "session_id": state.session_id,
        "participant_id": state.participant_id,
        "status": state.status,
        "started_at": state.started_at,
        "ended_at": state.ended_at,
        "stop_command": state.stop_command,
        "narration_status": state.narration_status,
        "capture": {"target_fps": capture_fps, "caption_interval_s": caption_interval_s},
        "files": {
            "frame_index": "frames/index.jsonl",
            "transcript": "transcript.jsonl",
            "captions": "captions.jsonl",
            "summary": "summary.md",
        },
        "counts": {
            "frames": state.frame_count,
            "transcripts": state.transcript_count,
            "captions": state.caption_count,
        },
        "hierarchy": [asdict(activity) for activity in state.activities],
    }


def _timeline_lines(state: CaptureSession) -> list[str]:
    lines = [
        "",
        "## Combined timeline",
        "",
        (
            "Narration (**Heard**, speech-to-text) and visual captions (**Saw**, "
            "model-generated) in time order. They can disagree or contain errors; "
            "inspect the frames before trusting either."
        ),
        "",
    ]
    if not state.timeline:
        lines.append("Waiting for narration or the first visual caption.")
        return lines
    # Heard sorts before Saw at the same instant; both timestamps are alignment hints.
    for timestamp_us, kind, text in sorted(state.timeline, key=lambda item: (item[0], item[1] != "Heard")):
        lines.append(f"- `{elapsed_hms(timestamp_us, state.started_at_us)}` **{kind}:** {_markdown(text)}")
    return lines


def summary(state: CaptureSession) -> str:
    """The human-readable view written as ``summary.md``."""
    lines = [
        f"# Capture {state.session_id}",
        "",
        f"Status: {state.status}  ",
        f"Started: {state.started_at}  ",
        f"Frames: {state.frame_count} · Transcripts: {state.transcript_count} · Captions: {state.caption_count}",
        "",
        "## Hierarchical visual summary",
        "",
        "| Level | Time range | Summary | Latest delta | Evidence |",
        "|---|---:|---|---|---|",
    ]
    if not state.activities:
        lines.append("| Activity | 00:00:00 | Waiting for the first visual caption |  |  |")
    for number, activity in enumerate(state.activities, start=1):
        t0 = state.started_at_us
        span = f"{elapsed_hms(activity.started_at_us, t0)}–{elapsed_hms(activity.ended_at_us, t0)}"
        lines.append(
            f"| **Activity {number}** | {span} | **{_markdown(activity.name)}** |  | "
            f"{activity.caption_count} captions |"
        )
        for phase in activity.phases:
            span = f"{elapsed_hms(phase.started_at_us, t0)}–{elapsed_hms(phase.ended_at_us, t0)}"
            lines.append(
                f"| ↳ Phase | {span} | **{_markdown(phase.name)}:** {_markdown(phase.summary)} | "
                f"{_markdown(phase.latest_delta)} | [frame {phase.evidence_frame_id}]({phase.evidence_path}) |"
            )
    lines.extend(_timeline_lines(state))
    lines.extend(
        (
            "",
            (
                "Frames and captions are source records; narration uses final transcripts from the shared voice "
                "runtime. `packet.json` is the current machine-readable index for coding agents."
            ),
            "",
        )
    )
    return "\n".join(lines)
