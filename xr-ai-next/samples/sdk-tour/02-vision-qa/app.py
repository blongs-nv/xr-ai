# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision Q&A: a conversation that can look through the camera.

SHAPE: one conversation with one TOOL. The language model decides when to
look; the tool asks the perception model about the newest frame. No router.

  turn -> conversation.answer -> (model calls `look`) -> perceive.ask(frame)
       -> answer spoken

test_app.py drives this app with fake models and checks what was said and
what was traced.
"""

from pathlib import Path

from pydantic import BaseModel
from xr_ai_kit import Conversation
from xr_ai_next import App, Hub, Participant, PerConnection, Voice, tool

HERE = Path(__file__).resolve().parent


class Settings(BaseModel):
    frame_max_age_s: float = 2.0
    frame_timeout_s: float = 3.0


app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
SETTINGS = app.settings(Settings)

hub = app.add(Hub())  # the device connection: who is here, and their camera
answer = app.role("answer_with_reasoning")
perceive = app.role("perceive")
voice = app.add(Voice(summarize=app.role("answer_lightweight")))


@tool
async def look(p: Participant, question: str) -> str:
    """Look through the user's camera and answer `question` about it."""
    frame = await p.frames.current(
        max_age_s=SETTINGS.frame_max_age_s, timeout_s=SETTINGS.frame_timeout_s
    )
    if frame is None:
        return "The camera is not streaming."
    return await perceive.ask(
        frame,
        question,
        system="Answer only from what is visible. Be brief.",
        max_tokens=96,
    )


conversation = app.add(
    Conversation(
        role=answer,
        output=voice,
        instructions="You help the user with what they see. Use `look` "
        "whenever the answer depends on the camera. Keep answers short.",
        tools=[look],
        context=[voice.dialogue],
    )
)
app.on_turn(conversation.answer)

if __name__ == "__main__":
    app.run()
