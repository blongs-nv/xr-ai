# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The capture component: the frame, caption and narration work of one capture, and sealing its packet.

See the package docstring for the packet and the wiring.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field
from xr_ai_app import Component, trace

from .packet import CaptureSession, SavedFrame, iso, now_us, packet, safe, session_stamp, summary, update_hierarchy

if TYPE_CHECKING:
    from xr_ai_app import Frame, Participant, Role, Sink

__all__ = ["DEFAULT_CAPTURE_MESSAGES", "Caption", "CaptureSession", "ProcedureCapture", "caption_prompt"]

logger = logging.getLogger(__name__)

DEFAULT_CAPTURE_MESSAGES: dict[str, str] = {
    "welcome": "Say start capture to start the recording and demonstrate your procedure. "
    "To finish recording, say stop capture.",
    "started": "Capture started.",
    "stopped": "Capture stopped.",
    "failed": "I could not start the capture.",
    "no_video": "I'm not getting any camera frames, so nothing is being recorded. Turn on the camera to record.",
}

_FIRST_FRAME_S = 5.0
"""A capture that has saved no frame this long after its start says ``no_video``."""


def caption_prompt() -> str:
    """The default captioning instructions (the ones the guide-generation evaluation was tuned with)."""
    return resources.files(__package__).joinpath("prompts/caption.txt").read_text(encoding="utf-8")


class Caption(BaseModel):
    """One structured caption of the current frame."""

    caption: str = Field(min_length=1, max_length=1000, description="What the CURRENT image shows.")
    delta: str = Field(min_length=1, max_length=1000, description="The change from the earlier image.")
    activity: str = Field(min_length=1, max_length=1000, description="The broad parent task.")
    phase: str = Field(min_length=1, max_length=1000, description="The current sub-action or state.")


def _caption_query(previous: dict[str, str] | None) -> str:
    query = (
        "You received exactly TWO images, in chronological order: FIRST=earlier, SECOND=current. "
        "Describe the SECOND image in caption. In delta use: Earlier: ... Current: ... Change: ... . "
        "Do not use Initial observation: there are two images. Then identify the broad parent activity "
        "and the specific current phase from the visual evidence."
        if previous is not None
        else "You received exactly ONE image: the current initial view. Describe that image in caption; "
        "set delta to Initial observation. Identify a supported broad activity and current phase/state."
    )
    if previous is not None:
        hints = {"activity": previous["activity"], "phase": previous["phase"]}
        query += "\nPrevious grouping labels (fallible naming hints, not visual evidence): " + json.dumps(hints)
    return query


@dataclass(frozen=True, slots=True)
class _Action:
    kind: Literal["start", "narrate", "stop"]
    text: str
    timestamp_us: int


class ProcedureCapture(Component):
    """Records each participant's demonstrations as evidence packets (see the module docstring).

    ``caption`` is a vision role asked with one or two frames and a structured :class:`Caption` reply; ``output``
    (optional) speaks ``messages`` (keys ``welcome``, ``started``, ``stopped``, ``failed``, ``no_video``; ``None`` or
    an empty string skips one). ``caption_system`` replaces the default captioning instructions.

    Frames come from ``p.frames.every``: streamed video, or with the camera on demand one picture per frame tick
    (``frames_per_s`` pictures a second; live video is what a recording is for). When no frame has come a few seconds
    after the start (the device gives none) it says ``no_video``, and keeps recording what comes.
    """

    def __init__(
        self,
        *,
        caption: Role,
        output: Sink | None = None,
        frames_per_s: float = 2.0,
        caption_every_s: float = 5.0,
        caption_timeout_s: float = 15.0,
        caption_system: str | None = None,
        messages: dict[str, str | None] | None = None,
    ) -> None:
        super().__init__()
        if frames_per_s <= 0 or caption_every_s <= 0 or caption_timeout_s <= 0:
            raise ValueError("frames_per_s, caption_every_s and caption_timeout_s must be positive")
        self._caption_role = caption
        self._output = output
        self._frames_per_s = frames_per_s
        self._caption_every_s = caption_every_s
        self._caption_timeout_s = caption_timeout_s
        self._caption_system = caption_system if caption_system is not None else caption_prompt()
        self._messages = {**DEFAULT_CAPTURE_MESSAGES, **(messages or {})}
        self._first_frame_s = _FIRST_FRAME_S
        self._queues: dict[str, asyncio.Queue[_Action]] = {}
        self._recording: dict[str, bool] = {}  # accepted, not yet applied: a start may wait behind a seal
        self._open: dict[str, CaptureSession] = {}
        self._sealed: dict[str, list[CaptureSession]] = {}

    # -- what the app calls ------------------------------------------------------------------------------

    def recording(self, p: Participant) -> bool:
        """Whether a capture is running for ``p`` (counting a start that is accepted but not yet applied)."""
        return self._recording.get(p.id, False)

    def start(self, p: Participant, *, timestamp_us: int | None = None) -> bool:
        """Start a capture; ``False`` when one is already running."""
        if self.recording(p):
            return False
        self._recording[p.id] = True
        self._queue(p).put_nowait(_Action("start", "", timestamp_us or now_us()))
        return True

    def narrate(self, p: Participant, text: str, *, timestamp_us: int | None = None) -> bool:
        """Add one narrated utterance to the running capture; ``False`` when none is running."""
        if not self.recording(p) or not text.strip():
            return False
        self._queue(p).put_nowait(_Action("narrate", text.strip(), timestamp_us or now_us()))
        return True

    def stop(self, p: Participant, *, said: str = "", timestamp_us: int | None = None) -> bool:
        """Stop the running capture and seal its packet; ``False`` when none is running."""
        if not self.recording(p):
            return False
        self._recording[p.id] = False
        self._queue(p).put_nowait(_Action("stop", said.strip(), timestamp_us or now_us()))
        return True

    async def settled(self, p: Participant) -> None:
        """Wait until every accepted start, narration and stop of ``p`` has been applied."""
        queue = self._queues.get(p.id)
        if queue is not None:
            await queue.join()

    def current(self, p: Participant) -> CaptureSession | None:
        """The running capture of ``p``."""
        return self._open.get(p.id)

    def sealed(self, p: Participant) -> list[CaptureSession]:
        """The captures of ``p`` sealed so far, oldest first."""
        return list(self._sealed.get(p.id, []))

    # -- lifecycle ---------------------------------------------------------------------------------------

    async def on_participant_enrolled(self, p: Participant) -> None:
        self._queue(p)
        await self._say(p, "welcome")

    async def on_participant_unenrolled(self, p: Participant) -> None:
        # The worker has been drained; apply what was accepted before the leave (a start no longer counts).
        queue = self._queues.pop(p.id, None)
        while queue is not None and not queue.empty():
            action = queue.get_nowait()
            if action.kind != "start":
                await self._apply(p, action)
        session = self._open.get(p.id)
        if session is not None:
            await self._seal(p, session, status="complete")
        self._recording.pop(p.id, None)

    def _queue(self, p: Participant) -> asyncio.Queue[_Action]:
        queue = self._queues.get(p.id)
        if queue is None:
            queue = self._queues[p.id] = asyncio.Queue()
            p.spawn(self._worker(p, queue), key="capture.actions")
        return queue

    async def _worker(self, p: Participant, queue: asyncio.Queue[_Action]) -> None:
        while True:
            action = await queue.get()
            try:
                await self._apply(p, action)
            except Exception:  # noqa: BLE001 - one failed action must not end the queue
                logger.exception("capture action %s failed for %s", action.kind, p.id)
            finally:
                queue.task_done()

    async def _apply(self, p: Participant, action: _Action) -> None:
        session = self._open.get(p.id)
        if action.kind == "start":
            if session is None:
                await self._start(p, action.timestamp_us)
        elif session is None:
            return
        elif action.kind == "narrate":
            await self._narrate(p, session, action)
        else:
            session.stop_command = {"text": action.text, "timestamp_us": action.timestamp_us}
            await self._say(p, "stopped")
            await self._seal(p, session, status="complete")

    async def _say(self, p: Participant, key: str) -> None:
        """Best effort: a failed announcement never prevents saving."""
        text = self._messages.get(key)
        if self._output is None or not text:
            return
        try:
            await p.say(text, to=self._output)
        except Exception:  # noqa: BLE001
            logger.exception("capture announcement %s failed for %s", key, p.id)

    # -- start and evidence ------------------------------------------------------------------------------

    @staticmethod
    def _name(session: CaptureSession, relative: str) -> str:
        return f"sessions/{session.session_id}/{relative}"

    async def _start(self, p: Participant, timestamp_us: int) -> None:
        session = CaptureSession(p.id, f"{session_stamp()}-{safe(p.id)[:48]}", iso(timestamp_us), timestamp_us)
        self._open[p.id] = session
        try:
            await p.artifacts.write_blob(self._name(session, "transcript.jsonl"), b"")
            await self._write_views(p, session)
        except Exception:  # noqa: BLE001 - a failed start leaves an incomplete packet and a fresh start possible
            logger.exception("capture start failed for %s", p.id)
            session.status = "incomplete"
            await self._seal(p, session, status="incomplete")
            self._recording[p.id] = False
            await self._say(p, "failed")
            return
        p.spawn(self._sample(p, session), key="capture.frames")
        p.spawn(self._expect_video(p, session), key="capture.video", supersede=True)
        trace.emit("procedure", "capture", phase="start", session_id=session.session_id)
        await self._say(p, "started")

    async def _narrate(self, p: Participant, session: CaptureSession, action: _Action) -> None:
        if not session.active:
            return
        session.transcript_count += 1
        session.timeline.append((action.timestamp_us, "Heard", action.text))
        entry = {
            "transcript_id": session.transcript_count,
            "timestamp_us": action.timestamp_us,
            "timestamp": iso(action.timestamp_us),
            "text": action.text,
        }
        try:
            await p.artifacts.write_jsonl(self._name(session, "transcript"), entry)
        except Exception:
            session.status, session.narration_status = "incomplete", "failed"
            raise

    async def _expect_video(self, p: Participant, session: CaptureSession) -> None:
        """Say so when no frame was saved soon after the start: a capture records streamed video only."""
        await asyncio.sleep(self._first_frame_s)
        if session.active and session.frame_count == 0:
            await self._say(p, "no_video")

    async def _sample(self, p: Participant, session: CaptureSession) -> None:
        """Save each distinct frame (at most ``frames_per_s``); start captioning with the first."""
        async for frame in p.frames.every(1.0 / self._frames_per_s):
            if not session.active:
                return
            try:
                await self._save_frame(p, session, frame)
            except Exception as exc:  # noqa: BLE001 - one failed frame does not end the capture
                logger.warning("frame capture failed for %s: %s", p.id, exc)
                await self._error(p, session, "capture_error", str(exc))

    async def _save_frame(self, p: Participant, session: CaptureSession, frame: Frame) -> None:
        key = (frame.timestamp_us, frame.sequence)
        if key == session.last_frame_key:
            return
        frame_id = session.frame_count + 1
        relative = f"frames/frame_{frame_id:06d}_{frame.timestamp_us}.jpg"
        session.frame_count, session.last_frame_key = frame_id, key
        session.latest_frame = SavedFrame(
            frame_id, frame.timestamp_us, relative, frame.sequence, frame.width, frame.height, frame
        )
        try:
            await p.artifacts.write_blob(self._name(session, relative), frame.image)
            entry = {
                "frame_id": frame_id,
                "timestamp_us": frame.timestamp_us,
                "timestamp": iso(frame.timestamp_us),
                "path": relative,
                "sequence": frame.sequence,
                "width": frame.width,
                "height": frame.height,
            }
            await p.artifacts.write_jsonl(self._name(session, "frames/index"), entry)
        except Exception:
            session.status = "incomplete"
            raise
        if frame_id == 1:
            p.spawn(self._caption_loop(p, session), key="capture.captions")

    async def _caption_loop(self, p: Participant, session: CaptureSession) -> None:
        while session.active:
            try:
                await self._caption_once(p, session)
            except Exception as exc:  # noqa: BLE001 - one failed caption does not end the capture
                logger.warning("frame caption failed for %s: %s", p.id, exc)
                await self._error(p, session, "caption_error", str(exc))
            await asyncio.sleep(self._caption_every_s)

    async def _caption_once(self, p: Participant, session: CaptureSession) -> None:
        frame, previous_frame = session.latest_frame, session.last_captioned_frame
        if frame is None or (previous_frame is not None and frame.frame_id == previous_frame.frame_id):
            return
        previous = dict(session.previous_caption) if session.previous_caption else None
        images = [item.frame for item in (previous_frame, frame) if item is not None and item.frame is not None]
        reply: Caption = await self._caption_role.ask(
            images,
            _caption_query(previous),
            response_format=Caption,
            system=self._caption_system,
            timeout_s=self._caption_timeout_s,
        )
        if not session.active:
            return
        caption = {k: v.strip() for k, v in reply.model_dump().items()}
        generated = now_us()
        session.caption_count += 1
        entry = {
            "caption_id": session.caption_count,
            "frame_id": frame.frame_id,
            "frame_timestamp_us": frame.timestamp_us,
            "frame_timestamp": iso(frame.timestamp_us),
            "generated_at_us": generated,
            "generated_at": iso(generated),
            "frame_path": frame.path,
            "delta_reference_frame_id": previous_frame.frame_id if previous_frame else None,
            "delta_reference_timestamp_us": previous_frame.timestamp_us if previous_frame else None,
            **caption,
        }
        session.previous_caption, session.last_captioned_frame = caption, frame
        update_hierarchy(session, entry)
        await p.artifacts.write_jsonl(self._name(session, "captions"), entry)

    async def _error(self, p: Participant, session: CaptureSession, kind: str, message: str) -> None:
        if not session.active:
            return
        try:
            await p.artifacts.write_jsonl(
                self._name(session, "errors"), {"timestamp_us": now_us(), "kind": kind, "message": message}
            )
        except Exception:  # noqa: BLE001
            logger.exception("capture error log failed for %s (%s)", p.id, kind)

    # -- sealing -----------------------------------------------------------------------------------------

    async def _write_views(self, p: Participant, session: CaptureSession) -> None:
        cfg: dict[str, Any] = {"capture_fps": self._frames_per_s, "caption_interval_s": self._caption_every_s}
        body = json.dumps(packet(session, **cfg), indent=2, ensure_ascii=False) + "\n"
        await p.artifacts.write_blob(self._name(session, "packet.json"), body.encode(), replace=True)
        await p.artifacts.write_blob(self._name(session, "summary.md"), summary(session).encode(), replace=True)

    async def _rebuild(self, p: Participant, session: CaptureSession) -> None:
        """Recompute counts, hierarchy and timeline from the written logs (a cancelled writer may skip memory)."""
        read = p.artifacts.read_jsonl
        session.frame_count = len(await read(self._name(session, "frames/index")))
        captions = await read(self._name(session, "captions"))
        session.caption_count = len(captions)
        session.activities.clear()
        session.timeline.clear()
        for entry in captions:
            update_hierarchy(session, entry)
        heard = await read(self._name(session, "transcript"))
        session.transcript_count = len(heard)
        session.timeline.extend((int(e["timestamp_us"]), "Heard", str(e["text"])) for e in heard)
        if session.narration_status != "failed":
            session.narration_status = "complete"

    async def _seal(self, p: Participant, session: CaptureSession, *, status: str) -> None:
        """Stop the frame and caption work, wait for the writes, rebuild from the logs, write the final packet."""
        if session.sealed:
            return
        await p.cancel("capture.frames")  # drained: no evidence after this point
        await p.cancel("capture.captions")
        await p.cancel("capture.video")
        session.active = False
        session.ended_at = session.ended_at or iso(now_us())
        if self._open.get(p.id) is session:
            del self._open[p.id]
        try:
            await p.artifacts.flush()
            await self._rebuild(p, session)
            if session.status != "incomplete":
                session.status = status
            await self._write_views(p, session)
        except Exception:  # noqa: BLE001
            session.status = "incomplete"
            logger.exception("capture finalization failed for %s", p.id)
            try:  # a failed view must never leave a packet advertised as complete
                await self._write_views(p, session)
            except Exception:  # noqa: BLE001
                logger.exception("incomplete packet seal failed for %s", p.id)
        finally:
            session.sealed = True
            self._sealed.setdefault(p.id, []).append(session)
        trace.emit(
            "procedure",
            "capture",
            phase="sealed",
            session_id=session.session_id,
            status=session.status,
            frames=session.frame_count,
            captions=session.caption_count,
            transcripts=session.transcript_count,
        )
