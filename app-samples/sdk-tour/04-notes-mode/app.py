# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notes mode: a Router, and a mode that holds the focus.

SHAPE: a ROUTER in front of two destinations. "Start taking notes" opens a
MODE: while it is on every turn goes to the notes first (no routing model
call), until "stop". A question during the mode is handed back to the
router, which answers it with the conversation; the mode stays on.

  turn -> Router: does a destination hold focus? (notes, while on)
            yes -> notes             no -> decide picks one
         -> notes asks decide: stop, content or other
            stop -> "Okay, notes stopped."   content -> take_note()
            other -> declined: the router picks again, without notes

Both destinations are kit helpers: `OneShotDestination` routes a plain
handler (the conversation's `answer`); `StartStopDestination` is a mode
started and stopped by asking, around a plain handler (`take_note`) that
knows nothing about routing. For anything else, build a `RouterDestination`
with its `admit`/`focused` hooks. The Router is built on the decision role's
`decide` (see 05-manual-dispatch for routing written by hand).
"""

from pathlib import Path

from xr_ai_app import (
    App,
    ContextLog,
    Hub,
    Participant,
    PerConnection,
    Turn,
    Voice,
)
from xr_ai_kit import (
    Ack,
    Conversation,
    OneShotDestination,
    Router,
    StartStopDestination,
)

HERE = Path(__file__).resolve().parent

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
app.add(Hub())  # the device connection: who is here
reasoning = app.role("answer_with_reasoning")
lightweight = app.role("answer_lightweight")
decide = app.role("decide")
voice = app.add(Voice(summarize=lightweight))

# The notes, as text the conversation can read (so "read my notes back" is
# a question for it). Participant scope: they outlive a reconnect when
# enrollment is persistent.
notes = ContextLog(
    "notes", scope="participant", budget_bytes=4_000, summarize=lightweight
)


async def take_note(turn: Turn, p: Participant) -> None:
    """One note: the turn's words. Nothing here about the mode."""
    await notes.publish(p, turn.text)
    await p.reply("Noted.", to=voice)


note_mode = StartStopDestination(
    "notes",
    "Note taking: start or stop a note session, or dictate a note.",
    take_note,
    decide=decide,
    output=voice,
    start="The user wants to start taking notes.",
    content="A statement to write down; not a request.",
    stop="The user wants to stop taking notes.",
    other="A question or request for the assistant, including hearing the "
    "notes back.",
    started="Okay, I'm taking notes.",
    stopped="Okay, notes stopped.",
)

conversation = app.add(
    Conversation(
        role=reasoning,
        output=voice,
        instructions="You are a helpful assistant. Be brief.",
        context=[voice.dialogue, notes],
    )
)
chat = OneShotDestination(
    "conversation",
    "Questions and chat. Knows the user's notes.",
    conversation.answer,
)

app.add(
    Router(
        decide=decide,
        destinations=[chat, note_mode],
        default=chat,
        history=voice.dialogue,
        output=voice,
        ack=Ack(role=lightweight, delay_s=1.5),
    )
)

if __name__ == "__main__":
    app.run()
