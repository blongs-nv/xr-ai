# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guided procedure: a spoken checklist the camera verifies.

SHAPE: a PROCEDURE (xr_ai_kit.sop). The procedure is a plain async function
that runs its steps in order; a SopRunner runs it for each participant who
starts it, and the user can say start, stop, skip or restart, or ask about
it. Each camera step judges a frame a second (perceive, then decide) until
two of the last three agree; beside it, an escalator of its own
(StepWithEscalator) watches what the step sees and, when the camera cannot
tell for a while, tells the user what to show, or asks them. Each step says
its own lines.

  xr_ai_kit.sop                     this file
  -------------------------------   ----------------------------------------
  SopRunner (component)             preflight(run): the steps and their order
  WatchStep, WaitStep (steps)       sop.yaml: the steps' words
  StepWithEscalator,                what each escalator reads (ESCALATION)
    ReasoningEscalator              the models each step uses
  controls: start | stop |
    skip | restart

The runner publishes a summary of where the run is to `preflight.context`;
the conversation reads it, so "what step am I on?" just works.
"""

from pathlib import Path

from xr_ai_app import App, Hub, PerConnection, Voice
from xr_ai_kit import Conversation, OneShotDestination, Router
from xr_ai_kit.sop import (
    EscalationSettings,
    EscalationWords,
    ReasoningEscalator,
    Run,
    SopRunner,
    StepWithEscalator,
    StepWords,
    WaitStep,
    WatchModels,
    WatchSettings,
    WatchStep,
    WatchWords,
    load_words,
)

HERE = Path(__file__).resolve().parent

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
hub = app.add(Hub())
perceive = app.role("perceive")
decide = app.role("decide")
answer_with_reasoning = app.role("answer_with_reasoning")
voice = app.add(Voice(summarize=app.role("answer_lightweight")))

WARM_UP_S = 90

# ---- the steps: words from sop.yaml, types and models here ----
SOP = HERE / "sop.yaml"
# perceive describes each frame; decide labels the description.
CAMERA_CHECK_MODELS = WatchModels(perceive=perceive, decide=decide)
# A look a second; done when two of the last three decided looks are yes.
WATCH = WatchSettings(required=2, window=3, interval_s=1.0)
# Each camera step gets its own escalator: when the camera cannot tell, it
# tells the user what to show, or asks them. Each reads what it needs:
# here the checklist's state ("run") and the recent dialogue.
ESCALATION = EscalationSettings(context=["run", "dialogue"])

bed_clear = StepWithEscalator(
    WatchStep(
        "bed_clear",
        load_words(SOP, WatchWords, key="bed_clear.step"),
        CAMERA_CHECK_MODELS,
        settings=WATCH,
    ),
    ReasoningEscalator(
        load_words(SOP, EscalationWords, key="bed_clear.escalation"),
        answer_with_reasoning,
        ESCALATION,
    ),
)
door_closed = StepWithEscalator(
    WatchStep(
        "door_closed",
        load_words(SOP, WatchWords, key="door_closed.step"),
        CAMERA_CHECK_MODELS,
        settings=WATCH,
    ),
    ReasoningEscalator(
        load_words(SOP, EscalationWords, key="door_closed.escalation"),
        answer_with_reasoning,
        ESCALATION,
    ),
)
warm_up = WaitStep(
    "warm_up",
    load_words(SOP, StepWords, key="warm_up.step"),
    duration_s=WARM_UP_S,
)
STEPS = (bed_clear, door_closed, warm_up)


async def preflight(run: Run) -> None:
    """The procedure: ordinary code. Each step says its own lines."""
    await run.step(bed_clear)
    await run.step(door_closed)
    await run.step(warm_up)


procedure = app.add(
    SopRunner(
        "preflight",
        preflight,
        title="Printer pre-flight",
        decide=decide,
        output=voice,
        # the steps ahead, for the conversation
        plan=[(step.id, step.words.does) for step in STEPS],
        escalation_context=[voice.dialogue],  # what escalators may read
    )
)
# Spoken answers; the checklist's state (in the context) is the only source
# for questions about it, so the model does not guess.
INSTRUCTIONS = (
    "You help with a 3D printer pre-flight checklist. Answer questions about "
    "the checklist (its steps, where the user is, how long is left) only "
    "from its state in your context; when the state does not say, say so "
    "instead of guessing. Your answer is spoken: one or two short sentences, "
    "no lists, tables or Markdown."
)
conversation = app.add(
    Conversation(
        role=answer_with_reasoning,
        output=voice,
        instructions=INSTRUCTIONS,
        context=[voice.dialogue, procedure.context],
    )
)
chat = OneShotDestination(
    "conversation",
    # Examples steer the router: questions here, commands to the procedure.
    "Questions about printing and about the checklist, also where the user "
    'is in it (it reads the checklist\'s state): "what step am I on?", '
    '"what\'s next?", "which steps are done?", "how long is left?", "what '
    'temperature for PLA?". Not commands to the checklist: not "start the '
    'checklist", "next step", "skip this", "stop".',
    conversation.answer,
)
routing = app.add(
    Router(
        decide=decide,
        destinations=[chat, procedure.control],
        default=chat,
        history=voice.dialogue,
        output=voice,
    )
)
app.on_turn(routing.handle)  # every turn goes to the router first

if __name__ == "__main__":
    app.run()
