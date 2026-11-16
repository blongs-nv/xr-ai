# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera monitor: an assistant that speaks up on its own.

SHAPE: a PROACTIVE component. Nothing the user says starts it: it starts a
background loop for every participant who joins, reads a camera frame
every second into a small schema, and speaks when a watched condition is
confirmed: said yes in enough looks in a row (a look that could not tell is
skipped, neither yes nor no). The conversation answers questions about what
it saw (it reads the monitor's context value) and can pause or resume it
(tools).

  join -> Monitor.on_participant_enrolled -> p.spawn(watch loop)
  watch loop: p.frames.every(1 s): a new streamed frame, or with the camera
              on demand one picture asked of the device, per tick
              -> look(): perceive.ask(response_format=Scene)
              -> a yes streak per condition (cannot_tell skipped)
              -> p.say(..., urgent=True)      (speaks without a turn)
              -> seen.set(p, "...")           (what the conversation reads)

What to watch is DATA (app.yaml `settings.conditions`): each condition is a
field of the schema the perception model fills, and a sentence to say.
"""

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, create_model
from xr_ai_app import (
    App,
    Component,
    ContextValue,
    Frame,
    Hub,
    Participant,
    PerConnection,
    Voice,
    clock,
    tool,
)
from xr_ai_kit import Conversation

HERE = Path(__file__).resolve().parent


class Condition(BaseModel):
    question: str  # asked of each frame, answered yes / no / cannot_tell
    say: str  # spoken once when it becomes true


class Settings(BaseModel):
    look_every_s: float = 1.0
    frame_max_age_s: float = 2.0
    required_looks: int = 3  # this many "yes" in a row (cannot_tell skipped)
    cooldown_s: float = 30.0  # then quiet this long for that condition
    conditions: dict[str, Condition] = {}


app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
SETTINGS = app.settings(Settings)

hub = app.add(Hub())
perceive = app.role("perceive")
lightweight = app.role("answer_lightweight")
voice = app.add(Voice(summarize=lightweight))

# One schema for every look: a field per condition, plus a one-line summary.
# Built from the settings, so a new condition needs no code.
Answer = Literal["yes", "no", "cannot_tell"]
Scene = create_model(
    "Scene",
    summary=(str, Field(description="One short sentence: what is visible.")),
    **{
        name: (Answer, Field(description=c.question))
        for name, c in SETTINGS.conditions.items()
    },
)

# What the monitor last saw: replaced on every look, read by the conversation.
seen = ContextValue("camera monitor", scope="session", budget_bytes=400)


class Monitor(Component):
    async def on_participant_enrolled(self, p: Participant) -> None:
        p.session.state.monitor_on = asyncio.Event()  # cleared while paused
        p.session.state.monitor_on.set()
        p.spawn(self.watch(p), key="monitor")

    async def watch(self, p: Participant) -> None:
        streak = dict.fromkeys(SETTINGS.conditions, 0)  # "yes" in a row
        quiet_until = dict.fromkeys(SETTINGS.conditions, 0.0)
        on = p.session.state.monitor_on
        while True:
            await on.wait()
            # While paused it leaves every(): with the camera on demand, no
            # pictures are asked for.
            async for frame in p.frames.every(SETTINGS.look_every_s):
                if not on.is_set():
                    break
                verdicts = await self.look(p, frame)
                for name, verdict in verdicts.items():
                    # "no" breaks the streak; a look that cannot tell keeps it.
                    if verdict == "yes":
                        streak[name] += 1
                    elif verdict == "no":
                        streak[name] = 0
                    ready = streak[name] >= SETTINGS.required_looks
                    if ready and clock.now() >= quiet_until[name]:
                        quiet_until[name] = clock.now() + SETTINGS.cooldown_s
                        streak[name] = 0
                        # Not an answer to anything: `say` takes the floor on
                        # the assistant's own initiative. `source` lets a later
                        # `withdraw` drop it if it is still queued.
                        await p.say(
                            SETTINGS.conditions[name].say,
                            to=voice,
                            urgent=True,
                            source=name,
                        )
            else:
                return  # the participant left: every() ended

    async def look(self, p: Participant, frame: Frame) -> dict[str, str]:
        """One look: each condition's verdict (yes, no or unknown)."""
        scene = await perceive.ask(
            frame,
            "Answer each field about this camera image.",
            response_format=Scene,
            max_tokens=160,
        )
        await seen.set(p, f"Last look: {scene.summary}")
        verdict = {"yes": "yes", "no": "no", "cannot_tell": "unknown"}
        return {
            name: verdict[getattr(scene, name)] for name in SETTINGS.conditions
        }


@tool
async def pause_monitor(p: Participant) -> str:
    """Stop watching the camera until the user asks to resume."""
    p.session.state.monitor_on.clear()
    return "Paused."


@tool
async def resume_monitor(p: Participant) -> str:
    """Start watching the camera again."""
    p.session.state.monitor_on.set()
    return "Watching again."


monitor = app.add(Monitor())
conversation = app.add(
    Conversation(
        role=app.role("answer_with_reasoning"),
        output=voice,
        instructions="You are a camera assistant. The camera monitor section "
        "says what you last saw. Pause or resume the monitor when asked.",
        tools=[pause_monitor, resume_monitor],
        context=[voice.dialogue, seen],
    )
)
app.on_turn(conversation.answer)

if __name__ == "__main__":
    app.run()
