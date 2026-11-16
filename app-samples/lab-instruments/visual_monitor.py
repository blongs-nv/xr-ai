# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The visual monitor: caption the view every few seconds and note changes.

One background task per participant (`p.spawn`), started with an optional
focus and stopped by the conversation's tools. Each look asks the VLM
(upstream's `monitor_prompt.txt`) for a caption and whether something
material changed since the previous caption. As upstream it never speaks:
every look is written to `monitor.jsonl` and kept for the
`recent_visual_history` tool, and the baseline and each change are added to
the monitoring history the conversation reads.
"""

import asyncio
import json
from collections import deque

from pydantic import BaseModel, ConfigDict, Field, model_validator
from records import MonitorRecord, Records
from xr_ai_app import Component, ContextLog, Participant, Role, clock

KEY = "visual-monitor"
DEFAULT_FOCUS = "important visual changes"


class VisualMonitor(Component):
    def __init__(
        self,
        vlm: Role,
        *,
        prompt: str,
        records: Records,
        history: ContextLog,
        every_s: float,
        keep: int,
        max_age_s: float,
        timeout_s: float,
    ) -> None:
        super().__init__()
        self._vlm = vlm
        self._prompt = prompt
        self._records = records
        self._history = history
        self._every_s = every_s
        self._keep = keep
        self._max_age_s = max_age_s
        self._timeout_s = timeout_s

    def running(self, p: Participant) -> bool:
        task = p.state.get(KEY)
        return task is not None and not task.done()

    def start(self, p: Participant, instruction: str) -> str:
        if self.running(p):
            return (
                "Background monitoring is already running. "
                f"Monitoring: {p.state.visual_focus}."
            )
        focus = instruction.strip().rstrip(".!? ") or DEFAULT_FOCUS
        p.state.visual_focus = focus
        p.state[KEY] = p.spawn(self._watch(p, focus), key=KEY, detached=True)
        return f"Background monitoring started. Monitoring: {focus}."

    async def stop(self, p: Participant) -> str:
        if not await p.cancel(KEY):
            return "Background monitoring is not running."
        return "Background monitoring stopped."

    def status(self, p: Participant) -> str:
        if self.running(p):
            return (
                "Background monitoring is running. "
                f"Monitoring: {p.state.visual_focus}."
            )
        return "Background monitoring is stopped."

    def recent(self, p: Participant, limit: int) -> list[MonitorRecord]:
        """The newest `limit` looks, oldest first (this session's)."""
        return list(p.state.get("visual_looks", ()))[-limit:]

    async def _watch(self, p: Participant, focus: str) -> None:
        looks = p.state.setdefault("visual_looks", deque(maxlen=self._keep))
        previous: str | None = None
        while True:
            record = await self._look(p, focus, previous)
            looks.append(record)
            await self._records.write(p, "monitor", record)
            if record.record_type == "baseline":
                await self._history.publish(
                    p, f"Visual monitor ({focus}): {record.caption}"
                )
            elif record.changed:
                await self._history.publish(
                    p, f"Visual monitor change: {record.summary}"
                )
            if record.record_type in ("baseline", "observation"):
                previous = record.caption
            await asyncio.sleep(self._every_s)

    async def _look(
        self, p: Participant, focus: str, previous: str | None
    ) -> MonitorRecord:
        now = clock.wall_us()
        frame = await p.frames.current(
            max_age_s=self._max_age_s, timeout_s=self._timeout_s
        )
        if frame is None:
            return MonitorRecord(
                timestamp_us=now,
                record_type="unavailable",
                error="No camera frame available — please try again.",
            )
        # Focus and caption are data under the prompt, never instructions.
        data = {"monitoring_focus": focus, "previous_caption": previous}
        try:
            text = str(
                await self._vlm.ask(
                    frame,
                    json.dumps(data, ensure_ascii=False),
                    system=self._prompt,
                )
            )
        except Exception as exc:  # noqa: BLE001 - recorded, as upstream
            return MonitorRecord(
                timestamp_us=now, record_type="error", error=str(exc)
            )
        try:
            look = parse_monitor_response(text, baseline=previous is None)
        except ValueError as exc:
            return MonitorRecord(
                timestamp_us=now,
                record_type="error",
                caption=text[:2000],
                error=f"invalid monitor response: {exc}",
            )
        return MonitorRecord(
            timestamp_us=now,
            record_type="baseline" if previous is None else "observation",
            caption=look.caption,
            changed=look.changed,
            summary=look.summary,
        )


class Look(BaseModel):
    model_config = ConfigDict(extra="forbid")
    caption: str = Field(min_length=1, max_length=2000)
    changed: bool
    summary: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _summary_for_a_change(self) -> "Look":
        if self.changed and not self.summary.strip():
            raise ValueError("changed observations require a summary")
        return self


def parse_monitor_response(text: str, *, baseline: bool) -> Look:
    """The VLM's JSON reply (upstream's parse_monitor_response): a baseline
    never counts as a change, and no change carries no summary."""
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("monitor response did not contain a JSON object")
    look = Look.model_validate(json.loads(candidate[start : end + 1]))
    if baseline or not look.changed:
        return look.model_copy(update={"changed": False, "summary": ""})
    return look
