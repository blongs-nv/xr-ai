# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guided procedure: a spoken checklist the camera verifies.

SHAPE: a PROCEDURE (xr_ai_kit.sop). The procedure is a plain async function
that runs its steps in order; a SopRunner runs it for each participant who
starts it, and the user can say start, stop, skip or restart, or ask about
it. Each camera step judges a frame a second (observe, then decide) until two
of the last three agree; when the camera cannot tell for a while, a side
escalation tells the user what to show, or asks them.

  xr_ai_kit.sop                  this file
  ---------------------------    -------------------------------------------
  SopRunner (component)          preflight(run): the steps and their order
  WatchStep, WaitStep (steps)    sop.yaml: the words (Wording)
  controls: start | stop |       the roles each camera step uses
    skip | restart

The runner publishes a summary of where the run is to `preflight.context`;
the conversation reads it, so "what step am I on?" just works.
"""

from pathlib import Path

from xr_ai_app import App, Hub, PerConnection, Voice
from xr_ai_kit import Conversation, OneShotDestination, Router
from xr_ai_kit.sop import Roles, Run, SopRunner, WatchSettings, Wording

HERE = Path(__file__).resolve().parent

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
hub = app.add(Hub())
perceive = app.role("perceive")
decide = app.role("decide")
reasoning = app.role("answer_with_reasoning")
voice = app.add(Voice(summarize=app.role("answer_lightweight")))

WARM_UP_S = 90

# ---- the steps: words from sop.yaml, types and settings here ----
WORDS = Wording.load(HERE / "sop.yaml")
# observe describes a frame, decide labels it, reason runs the escalation.
ROLES = Roles(observe=perceive, decide=decide, reason=reasoning)
# A look a second; done when two of the last three decided looks are yes.
WATCH = WatchSettings(required=2, window=3, interval_s=1.0)

bed_clear = WORDS.watch("bed_clear", ROLES, WATCH)
door_closed = WORDS.watch("door_closed", ROLES, WATCH)
warm_up = WORDS.wait("warm_up", WARM_UP_S)


async def preflight(run: Run) -> None:
    """The procedure: ordinary code. A skipped step returns None."""
    if await run.step(bed_clear):
        await run.say(WORDS.line("bed_clear"))
    if await run.step(door_closed):
        await run.say(WORDS.line("door_closed"))
    if await run.step(warm_up):
        await run.say(WORDS.line("warmed_up"))


procedure = app.add(
    SopRunner(
        "preflight",
        preflight,
        title="Printer pre-flight",
        decide=decide,
        output=voice,
        plan=WORDS.plan(),  # the steps ahead, for the conversation
        step_context=[voice.dialogue],  # what the escalation also reads
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
