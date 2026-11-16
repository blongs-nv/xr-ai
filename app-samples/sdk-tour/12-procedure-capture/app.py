# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Procedure capture: record a demonstration as the evidence for a guide.

SHAPE: a RECORDER. The user says "start capture", does the procedure while
talking through it, and says "stop capture". Nothing is answered in between:
the app saves frames, captions them, keeps the narration, and writes an
evidence packet. A packet is what a guide is written from (the guide
generation skills of agent-samples/procedure-capture-guide read this exact
format); 10-guided-procedure then guides someone through the result.

  "start capture"   -> capture.start(p)      -> "Capture started."
  narration         -> capture.narrate(p, ...)   (no reply)
  the camera        -> frames twice a second; a caption every 5 s:
                       caption, change since the last one, activity, phase
                       (live video; with the camera on demand, one picture
                       per frame, twice a second)
  "stop capture"    -> capture.stop(p)       -> "Capture stopped." + packet
  leaving           -> an open capture is sealed

  artifacts/<participant>/sessions/<session-id>/
    packet.json  summary.md  frames/  captions.jsonl  transcript.jsonl

ProcedureCapture comes from the procedures kit (xr-ai-procedures). Its
start/stop/narrate return at once and are applied in order, so a turn that
is superseded never cuts a capture short.
"""

import re
from pathlib import Path

from pydantic import BaseModel
from xr_ai_app import App, Hub, Participant, PerConnection, Turn, Voice
from xr_ai_procedures import ProcedureCapture

HERE = Path(__file__).resolve().parent


class Settings(BaseModel):
    frames_per_s: float = 2.0
    caption_every_s: float = 5.0


app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
SETTINGS = app.settings(Settings)

hub = app.add(Hub())  # the device connection: who is here, and their camera
voice = app.add(Voice())
capture = app.add(
    ProcedureCapture(
        caption=app.role("perceive"),
        output=voice,  # speaks the welcome, "Capture started.", "stopped."
        frames_per_s=SETTINGS.frames_per_s,
        caption_every_s=SETTINGS.caption_every_s,
    )
)

COMMAND = re.compile(r"\s*(start|stop)\s+capture[.!?]*\s*", re.IGNORECASE)


async def on_turn(turn: Turn, p: Participant) -> None:
    command = COMMAND.fullmatch(turn.text)
    if command and command[1].lower() == "start":
        capture.start(p, timestamp_us=turn.timestamp_us)
    elif command:
        capture.stop(p, said=turn.text, timestamp_us=turn.timestamp_us)
    else:
        # Narration while capturing; anything else gets no reply.
        capture.narrate(p, turn.text, timestamp_us=turn.timestamp_us)


app.on_turn(on_turn)

if __name__ == "__main__":
    app.run()
