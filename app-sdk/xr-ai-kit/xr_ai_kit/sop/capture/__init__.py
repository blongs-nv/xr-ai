# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture: record a demonstrated procedure as an evidence packet, the input for writing a guide.

While a capture runs, :class:`ProcedureCapture` saves the participant's camera frames (``frames_per_s``), captions
the newest frame against the previous captioned one with a vision role (every ``caption_every_s``: a caption, the
change since, and a broad activity with its current phase), and keeps the narration it is given. Stopping seals the
packet under the participant's artifacts, ``sessions/<session-id>/``:

==========================  ========================================================================================
``packet.json``             the index: status, times, counts, the activity/phase hierarchy
``summary.md``              the same for people, plus a timeline of what was heard and seen
``frames/index.jsonl``      one line per saved frame (``frames/frame_<id>_<timestamp>.jpg``)
``captions.jsonl``          one line per caption
``transcript.jsonl``        one line per narrated utterance
``errors.jsonl``            missing frames, failed captions
==========================  ========================================================================================

It is the format of ``agent-samples/procedure-capture-guide``, whose ``guide-generation-skills`` turn a packet into
a guide; a :class:`~xr_ai_kit.sop.SopRunner` then guides someone through it.

Wiring::

    capture = app.add(ProcedureCapture(caption=app.role("perceive"), output=voice))

    async def on_turn(turn, p):
        if turn.text == "start capture": capture.start(p)
        elif turn.text == "stop capture": capture.stop(p, said=turn.text)
        else: capture.narrate(p, turn.text, timestamp_us=turn.timestamp_us)

``start``, ``stop`` and ``narrate`` return at once: a participant's actions are applied in order by one worker, so a
turn that is superseded never cuts a capture short, and a stop always seals what came before it. ``await
capture.settled(p)`` waits for them. A participant who leaves has their open capture sealed. The app needs an
``artifacts_dir``.
"""

from __future__ import annotations

from .packet import CaptureSession
from .recorder import DEFAULT_CAPTURE_MESSAGES, Caption, ProcedureCapture, caption_prompt

__all__ = ["DEFAULT_CAPTURE_MESSAGES", "Caption", "CaptureSession", "ProcedureCapture", "caption_prompt"]
