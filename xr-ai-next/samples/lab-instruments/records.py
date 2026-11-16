# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable output: upstream's JSONL files, records and fields.

Each participant's files go to `<artifacts_dir>/<participant>/<session>/`
(`p.artifacts`): `monitor.jsonl` (visual monitor), `instrument-monitoring
.jsonl` (changes, lost instruments, state snapshots) and `foreground.jsonl`
(each turn: query, response, tools used). As upstream, every file starts with
a `session` line and ends with a `session_end` line when the participant
leaves.
"""

import functools
from collections.abc import AsyncIterable, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_next import (
    Component,
    Participant,
    Voice,
    current_turn,
    tool,
)
from xr_ai_tools.marker_tracking import MarkerType


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MonitorRecord(Record):
    timestamp_us: int = Field(ge=0)
    record_type: Literal["baseline", "observation", "unavailable", "error"]
    caption: str = ""
    changed: bool = False
    summary: str = ""
    error: str = ""


class InstrumentState(Record):
    marker_type: MarkerType
    marker_id: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    meter_reading: str = Field(min_length=1)
    first_seen_us: int = Field(ge=0)
    last_seen_us: int = Field(ge=0)
    tracking: bool = True


class InstrumentChange(Record):
    event_type: Literal["change"] = "change"
    timestamp_us: int = Field(ge=0)
    change_type: Literal["discovered", "reading_changed"]
    marker_type: MarkerType
    marker_id: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    previous_reading: str | None = None
    meter_reading: str = Field(min_length=1)
    last_seen_us: int = Field(ge=0)


class InstrumentLost(Record):
    event_type: Literal["lost"] = "lost"
    timestamp_us: int = Field(ge=0)
    marker_type: MarkerType
    marker_id: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    meter_reading: str = Field(min_length=1)
    last_seen_us: int = Field(ge=0)


class InstrumentStateSnapshot(Record):
    event_type: Literal["state"] = "state"
    timestamp_us: int = Field(ge=0)
    instruments: list[InstrumentState] = Field(default_factory=list)


class ForegroundRecord(Record):
    timestamp_us: int = Field(ge=0)
    query: str = Field(min_length=1)
    response: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)


class Records(Component):
    """Writes each participant's records under one session directory."""

    async def on_participant_enrolled(self, p: Participant) -> None:
        p.state.session_dir = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        p.state.opened_files = set()

    async def write(self, p: Participant, name: str, record: Record) -> None:
        """Append `record` to `<name>.jsonl`, after its `session` line."""
        if name not in p.state.opened_files:
            p.state.opened_files.add(name)
            await self._line(p, name, "session")
        await p.artifacts.write_jsonl(self.path(p, name), record)

    def path(self, p: Participant, name: str) -> str:
        """The artifact name of `<name>.jsonl` for `p.artifacts`."""
        return f"{p.state.session_dir}/{name}"

    async def on_participant_unenrolled(self, p: Participant) -> None:
        for name in sorted(p.state.get("opened_files", ())):
            await self._line(p, name, "session_end")

    async def _line(self, p: Participant, name: str, kind: str) -> None:
        line = {
            "type": kind,
            "timestamp": datetime.now(UTC).isoformat(),
            "participant_id": p.id,
        }
        await p.artifacts.write_jsonl(self.path(p, name), line)


# -- what foreground.jsonl needs from a turn ---------------------------------


class AnswerVoice(Voice):
    """`Voice` that also keeps the turn's answer (the conversation's reply)."""

    async def reply(
        self, p: Participant, text: str | AsyncIterable[str]
    ) -> str:
        spoken = await super().reply(p, text)
        if spoken.strip() and (turn := current_turn()) is not None:
            turn.state.answer = spoken
        return spoken


def lab_tool(
    name: str, description: str, *, return_direct: bool = False
) -> Callable[[Any], Any]:
    """`@tool` under upstream's name, noting each use on the turn."""

    def mark(function: Callable[..., Awaitable[Any]]) -> Any:
        @functools.wraps(function)
        async def run(**arguments: Any) -> Any:
            if (turn := current_turn()) is not None:
                turn.state.setdefault("tools", []).append(name)
            return await function(**arguments)

        return tool(
            name=name, description=description, return_direct=return_direct
        )(run)

    return mark
