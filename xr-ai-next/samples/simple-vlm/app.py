# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple VLM: ask about what the camera sees, by voice or by typing.

SHAPE: one handler, the core only (no kit). The question and the current
camera frame go to the vision model; its answer streams back, spoken and
shown in the web client.

  turn -> p.frames.current() -> vlm.stream(frame, question) -> spoken

When the client is not streaming video, `current()` asks it for one picture
(the web client's on-demand camera mode). The SDK waits for the model
servers before it reports ready, and shows "processing" while a turn is
being answered.
"""

from pathlib import Path

from xr_ai_next import App, Hub, Participant, PerConnection, Turn, Voice

HERE = Path(__file__).resolve().parent
SYSTEM = (HERE / "prompts" / "system.txt").read_text(encoding="utf-8")

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
app.add(Hub())  # the device connection: who is here, and their camera
vlm = app.role("vlm")  # roles are the entry names of yaml/models.json
voice = app.add(Voice(tts=app.role("tts")))


@app.on_turn
async def answer(turn: Turn, p: Participant) -> None:
    frame = await p.frames.current(max_age_s=5.0, timeout_s=5.0)
    if frame is None:
        await voice.reply(p, "I can't see anything right now.")
        return
    await voice.reply(p, vlm.stream(frame, turn.text, system=SYSTEM))


def main() -> None:
    """`simple_vlm_app`: the launcher passes --config and --ready-file."""
    app.run()


if __name__ == "__main__":
    main()
