# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared room: several people, one shared checklist.

SHAPE: MULTI-PARTY. Each person has their own voice, dialogue and session;
the checklist and the room log are APP-scoped, shared by everyone. When one
person checks an item, the others hear about it.

  Ana: "the valve is closed" -> checklist tool -> app.state.checklist
       -> room.publish("Ana checked: valve closed")   (everyone's context)
       -> every OTHER participant: p.say("Ana checked the valve.")

Three state lifetimes, one more than a single-user app:
  app.state          shared by everyone, as long as the app runs
  p.state            one person's (their enrollment)
  p.session.state    one person's connection
"""

from pathlib import Path

from xr_ai_app import (
    App,
    ContextLog,
    ContextValue,
    Hub,
    Participant,
    Persistent,
    Voice,
    tool,
)
from xr_ai_kit import Conversation

HERE = Path(__file__).resolve().parent

ITEMS = ["valve closed", "power off", "area clear", "tools stowed"]

# Persistent enrollment: a person who drops and reconnects within a minute is
# the same participant (same p.state).
app = App.from_config(
    HERE / "app.yaml", enrollment=Persistent(keep_for_s=60, admit=None)
)
app.add(Hub())  # the device connection: who is here
lightweight = app.role("answer_lightweight")
voice = app.add(Voice(summarize=lightweight))

app.state.checklist = dict.fromkeys(ITEMS, "")  # item -> who checked it

# App scope: one log and one value for the whole room.
room = ContextLog(
    "room", scope="app", budget_bytes=3_000, summarize=lightweight
)
checklist = ContextValue("checklist", scope="app", budget_bytes=600)


async def publish_checklist() -> None:
    lines = [
        f"- {item}: {who and f'done by {who}' or 'open'}"
        for item, who in app.state.checklist.items()
    ]
    await checklist.set(None, "\n".join(lines))


@tool
async def check_item(p: Participant, item: str) -> str:
    """Mark a checklist item as done. `item` is one of the checklist items."""
    if item not in app.state.checklist:
        return f"No such item. Items: {', '.join(ITEMS)}."
    app.state.checklist[item] = p.name
    await room.publish(None, f"{p.name} checked: {item}")
    await publish_checklist()
    for other in app.participants:
        if other is not p:
            await other.say(f"{p.name} checked {item}.", to=voice)
    return "Checked."


conversation = app.add(
    Conversation(
        role=app.role("answer_with_reasoning"),
        output=voice,
        instructions="You coordinate a team working through a safety "
        "checklist. Check items when someone reports them done. Be brief.",
        tools=[check_item],
        # The person's own dialogue, then what the whole room shares.
        context=[voice.dialogue, checklist, room],
    )
)
app.on_turn(conversation.answer)


@app.on_start
async def start() -> None:
    await publish_checklist()


if __name__ == "__main__":
    app.run()
