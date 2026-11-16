# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracing: seeing what an app did, and why.

SHAPE: hello-voice plus OBSERVABILITY. Every app emits trace events with no
code: each turn, routing decision, model call (role, latency, tokens, the
structured answer), tool call, context write, spoken line, spawned task and
delegated job. Every event carries ids that tie it to its cause:

  session  participant  turn  task  parent  kind  name  data  t  duration_s

so a line spoken by a background task 40 s later still points back to the
turn that started it. Where events go is configuration (app.yaml `trace:`),
or code: `app.trace.add(sink)`. A sink is any callable taking an `Event`.

This sample adds two things by hand:
  - a SPAN around its own work, so it shows up nested in the trace;
  - a SINK that warns when a turn is slow to start speaking.
"""

import logging
from pathlib import Path

from xr_ai_app import App, Hub, Participant, PerConnection, Turn, Voice
from xr_ai_app.trace import Event
from xr_ai_kit import Conversation

HERE = Path(__file__).resolve().parent
log = logging.getLogger("tracing-sample")

# app.yaml:  trace: {jsonl: traces/}   -> one JSON line per event, per day
app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
app.add(Hub())  # the device connection: who is here
answer = app.role("answer_lightweight")
voice = app.add(Voice(summarize=answer))
conversation = app.add(
    Conversation(
        role=answer,
        output=voice,
        instructions="Be brief.",
        context=[voice.dialogue],
    )
)


def warn_slow_first_audio(event: Event) -> None:
    """A sink: called for every event, must not block."""
    if event.kind == "first_output" and event.data["after_turn_s"] > 2.0:
        log.warning(
            "slow turn %s: %.1fs to first audio",
            event.turn,
            event.data["after_turn_s"],
        )


app.trace.add(warn_slow_first_audio)


@app.on_turn
async def handle(turn: Turn, p: Participant) -> None:
    # A span nests everything inside it (the model call, the reply) under
    # "spell-check" in the trace, with its duration.
    async with app.trace.span("spell-check", text_len=len(turn.text)):
        cleaned = turn.text.strip()
    await conversation.answer(turn.replace(text=cleaned), p)


if __name__ == "__main__":
    app.run()
