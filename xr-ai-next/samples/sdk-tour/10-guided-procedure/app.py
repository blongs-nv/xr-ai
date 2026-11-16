# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guided procedure: a spoken checklist the camera verifies.

SHAPE: a PROCEDURE (xr_ai_procedures). A state machine walks the user
steps; each camera step is a structured read plus a judge in plain code; the
user can say start, stop, next, back, or tell it facts. This is a small
tea guide, with no retrieval.

  procedures                     this file
  ---------------------------    -------------------------------------------
  GuidedProcedure (component)    the steps, their schemas and judges
  read_check, Timer (steps)      sop.yaml: the words and the order
  Watching (debounce policy)
  controls: start | stop | next | back | info, gated by state

The procedure publishes a summary of where it is to `procedure.channel`; the
conversation reads it, so "what step am I on?" just works.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from xr_ai_kit import Conversation, OneShotDestination, Router
from xr_ai_next import App, Hub, PerConnection, Voice
from xr_ai_procedures import GuidedProcedure, Sop, StepDef, Timer
from xr_ai_procedures.perception import Watching, read_check

HERE = Path(__file__).resolve().parent

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
hub = app.add(Hub())
perceive = app.role("perceive")
decide = app.role("decide")
reasoning = app.role("answer_with_reasoning")
voice = app.add(Voice(summarize=app.role("answer_lightweight")))

WATCHING = Watching(required_positives=2, window=3, stuck_after_s=20)
WARM_UP_S = 90


# ---- the camera steps: what to read, and how to judge it ----
class Bed(BaseModel):
    seen: str = Field(description="One short sentence about the print bed.")
    state: Literal["clear", "objects_on_it", "cannot_tell"]


class Door(BaseModel):
    seen: str = Field(description="One short sentence about the door.")
    state: Literal["closed", "open", "cannot_tell"]


def camera_step(question: str, reads: type[BaseModel], ok: str):
    def build(p, run, spec):
        step = read_check(
            frames=p.frames,
            perceive=perceive,
            question=question,
            reads=reads,
            judge=lambda r: {ok: "yes", "cannot_tell": "unknown"}.get(
                r.state, "no"
            ),
            text=lambda r: r.seen,
            watching=WATCHING,
        )
        return step, ()  # the key: nothing the user says changes this step

    return build


STEPS = [
    StepDef(
        "bed_clear",
        camera_step("Is the print bed clear?", Bed, "clear"),
    ),
    StepDef(
        "door_closed",
        camera_step("Is the printer door closed?", Door, "closed"),
    ),
    StepDef("warm_up", lambda p, run, spec: (Timer(WARM_UP_S), ())),
]


def lengths(run):
    """Lines for the summary the conversation reads: how long steps take."""
    if run.index < len(STEPS) - 1:  # before the warm-up
        return [
            f"warm_up (a {WARM_UP_S}-second timer) has not started; it starts "
            "after the camera checks, which take as long as the user needs. "
            "No time is running yet."
        ]
    return [f"warm_up is a {WARM_UP_S}-second timer."]


procedure = app.add(
    GuidedProcedure(
        sop=Sop.load(HERE / "sop.yaml"),
        steps=STEPS,
        output=voice,
        decide=decide,
        reason=reasoning,
        describe=lengths,
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
        context=[voice.dialogue, procedure.channel],
    )
)
chat = OneShotDestination(
    "conversation",
    # Examples steer the router: questions here, commands to the procedure.
    "Questions about printing and about the checklist, also where the user "
    'is in it (it reads the checklist\'s state): "what step am I on?", '
    '"what\'s next?", "which steps are done?", "how long is left?", "what '
    'temperature for PLA?". Not commands to the checklist: not "start the '
    'checklist", "next step", "skip this", "go back", "stop".',
    conversation.answer,
)
app.add(
    Router(
        decide=decide,
        destinations=[chat, procedure.control],
        default=chat,
        history=voice.dialogue,
        output=voice,
    )
)

if __name__ == "__main__":
    app.run()
