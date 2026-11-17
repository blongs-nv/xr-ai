# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tea making: guidance through a procedure that watches the camera.

SHAPE: a procedure (procedure.py, xr_ai_kit.sop) next to a conversation. A
Router asks Clef (`decide`) where each turn goes:

  tea           only the guide's controls: start, stop, skip a step,
                restart (SopRunner.control); anything else passes on
  observation   what the camera shows now, asked of the omni model
  conversation  everything else, including every question about the guide:
                it reads the guide's state (tea.context) and the dialogue

Starting the guide starts a run that outlives the turn: a question, or an
interruption, never cancels it. Each step watches the camera on its own
clock (identify, verify, timer); the runner speaks each instruction and
result, and the conversation answers from the same state. Roles are the
jobs, as in the teaching samples: perceive (the omni model on images, no
reasoning: the per-frame loop), answer_with_reasoning (the same model,
reasoning: answers, recommendations, escalation), answer_lightweight (no
reasoning: quick acknowledgements), decide (Clef: routing, controls and
every frame's verdict) and speech_out (Pocket TTS).
"""

from pathlib import Path

from procedure import Steps, TeaSettings, make_tea
from pydantic import BaseModel, Field
from xr_ai_app import App, Hub, Participant, PerConnection, Turn, Voice
from xr_ai_app.context import render
from xr_ai_kit import Ack, Conversation, OneShotDestination, Router
from xr_ai_kit.sop import SopRunner

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"


class Settings(BaseModel):
    frame_max_age_s: float = 3.0
    frame_timeout_s: float = 5.0
    dialogue_budget_bytes: int = 6_000
    routing_history_budget_bytes: int = 6_000
    tea: TeaSettings = Field(default_factory=TeaSettings)


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8").strip()


app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
SETTINGS = app.settings(Settings)
app.add(Hub())  # the device connection: who is here, and their camera
perceive = app.role("perceive")
answer = app.role("answer_with_reasoning")
decide = app.role("decide")
voice = app.add(
    Voice(
        tts=app.role("speech_out"),
        summarize=answer,
        dialogue_budget_bytes=SETTINGS.dialogue_budget_bytes,
    )
)

tea = app.add(
    SopRunner(
        "tea",
        make_tea(
            observe=perceive,
            answer=answer,
            decide=decide,
            steps=Steps.load(HERE / "steps.yaml"),
            settings=SETTINGS.tea,
            situation=lambda p: situation(p),  # defined below
        ),
        title="Tea guidance",
        decide=decide,
        output=voice,
        plan=[
            ("identify", "show the tea's label so its settings can be read"),
            ("fill_water", "fill the kettle with fresh water"),
            ("heat_water", "heat the water to the recommended temperature"),
            ("start_steeping", "put the tea in the hot water"),
            ("steep", "wait while the timer runs"),
        ],
        description=(
            "Only explicit controls of the tea-making guide: start, stop, "
            "skip the current step, restart. Not questions about tea, the "
            "steps, the settings, progress or timers."
        ),
    )
)


async def situation(p: Participant) -> str:
    """What is going on, for the side escalation: the guide's state (with
    the running check's latest report) and the recent dialogue."""
    return await render(
        p,
        [tea.context, voice.dialogue],
        budget_bytes=SETTINGS.dialogue_budget_bytes,
    )


conversation = app.add(
    Conversation(
        role=answer,
        output=voice,
        instructions=prompt("conversation"),
        context=[tea.context, voice.dialogue],
    )
)
chat = OneShotDestination(
    "conversation",
    "General conversation and every question about the tea guide: what to "
    "do, the current step, progress, settings, timers, earlier "
    "observations. Does not look at the camera and does not start, stop or "
    "skip anything.",
    conversation.answer,
)


async def look(turn: Turn, p: Participant) -> None:
    """Answer about the current camera frame (or a picture taken on demand)."""
    frame = await p.frames.current(
        max_age_s=SETTINGS.frame_max_age_s, timeout_s=SETTINGS.frame_timeout_s
    )
    if frame is None:
        await p.reply("I couldn't get a camera frame. Please try again.")
        return
    await p.reply(perceive.stream(frame, turn.text, system=prompt("view")))


view = OneShotDestination(
    "observation",
    "Questions that need the camera right now: describe or read what it "
    "shows (objects, labels, displays, the cup or kettle in view). Not "
    "remarks, opinions or statements about things in view.",
    look,
)

routing = app.add(
    Router(
        decide=decide,
        destinations=[tea.control, view, chat],
        default=chat,
        history=voice.dialogue,
        history_budget_bytes=SETTINGS.routing_history_budget_bytes,
        output=voice,
        # The conversation reasons before it answers; a quick line covers
        # the wait when nothing was said for a moment.
        ack=Ack(role=app.role("answer_lightweight"), delay_s=1.5),
    )
)
app.on_turn(routing.handle)  # every turn goes to the router first


def main() -> None:
    """`tea_making_app`: the launcher passes --config and --ready-file."""
    app.run()


if __name__ == "__main__":
    main()
