# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hello voice: the smallest spoken assistant.

SHAPE: one component answers every turn. There is no router: `app.on_turn`
names the one handler, and the conversation's `answer` is that handler.

  user speaks -> Voice (speech to text) -> app.on_turn -> conversation.answer
              -> p.reply(..., to=voice) -> Voice (text to speech)

The voice component keeps the dialogue (`voice.dialogue`, a context log), and
the conversation reads it, so the assistant remembers earlier turns.
"""

from pathlib import Path

from xr_ai_kit import Conversation
from xr_ai_next import App, Hub, PerConnection, Voice

HERE = Path(__file__).resolve().parent

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
app.add(Hub())  # the device connection: who is here

# Model roles: models.json says which model fills each one.
answer = app.role("answer_lightweight")

# Speech in and out. Which speech models, and when to listen, is app.yaml's
# `voice:` section. `summarize` compacts the dialogue log when it is full.
voice = app.add(Voice(summarize=answer, dialogue_budget_bytes=6_000))

conversation = app.add(
    Conversation(
        role=answer,
        output=voice,
        instructions="You are a friendly assistant. Answer in one or two "
        "short spoken sentences.",
        context=[voice.dialogue],
    )
)

# Every turn goes to the conversation. (Swap in a Router when there is more
# than one thing to send turns to: see 04-notes-mode.)
app.on_turn(conversation.answer)

if __name__ == "__main__":
    app.run()
