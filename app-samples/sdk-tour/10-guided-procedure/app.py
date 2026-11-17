# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guided procedure: a spoken checklist the camera verifies.

SHAPE: a PROCEDURE (xr_ai_kit.sop). The procedure is a plain async function
that runs its steps in order; a SopRunner runs it for each participant who
starts it, and the user can say start, stop, skip or restart, or ask about
it. Each camera step judges a frame a second (observe, then decide) until two
of the last three agree; beside it, an escalator of its own (Escalated)
watches what the step sees and, when the camera cannot tell for a while,
tells the user what to show, or asks them. Each step says its own lines.

  xr_ai_kit.sop                  this file
  ---------------------------    -------------------------------------------
  SopRunner (component)          preflight(run): the steps and their order
  WatchStep, WaitStep (steps)    sop.yaml: the steps' words (StepDefinitions)
  Escalated, ReportEscalator     what each escalator reads (HELP)
  controls: start | stop |       the models each step uses
    skip | restart

The runner publishes a summary of where the run is to `preflight.context`;
the conversation reads it, so "what step am I on?" just works.
"""

from pathlib import Path

from xr_ai_app import App, Hub, PerConnection, Voice
from xr_ai_kit import Conversation, OneShotDestination, Router
from xr_ai_kit.sop import (
    Escalated,
    EscalationSettings,
    ReportEscalator,
    Run,
    SopRunner,
    StepDefinitions,
    WaitStep,
    WatchSettings,
    WatchStep,
)

HERE = Path(__file__).resolve().parent

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
hub = app.add(Hub())
perceive = app.role("perceive")
decide = app.role("decide")
reasoning = app.role("answer_with_reasoning")
voice = app.add(Voice(summarize=app.role("answer_lightweight")))

WARM_UP_S = 90

# ---- the steps: words from sop.yaml, types and models here ----
DEFINITIONS = StepDefinitions.load(HERE / "sop.yaml")
# A look a second; done when two of the last three decided looks are yes.
WATCH = WatchSettings(required=2, window=3, interval_s=1.0)

# Each camera step gets its own escalator: when the camera cannot tell, it
# tells the user what to show, or asks them. Each reads what it needs:
# here the checklist's state ("run") and the recent dialogue.
HELP = EscalationSettings(context=["run", "dialogue"])


def camera_step(step_id: str) -> Escalated:
    """A watch step (perceive describes a frame, decide labels it) with its
    escalator beside it (reasoning)."""
    definition = DEFINITIONS[step_id]
    step = WatchStep.define(
        definition, observe=perceive, decide=decide, settings=WATCH
    )
    return Escalated(step, ReportEscalator.define(definition, reasoning, HELP))


bed_clear = camera_step("bed_clear")
door_closed = camera_step("door_closed")
warm_up = WaitStep.define(DEFINITIONS["warm_up"], WARM_UP_S)
DEFINITIONS.check_read()  # every section of sop.yaml is used


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
        plan=DEFINITIONS.plan(),  # the steps ahead, for the conversation
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
        role=reasoning,
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
