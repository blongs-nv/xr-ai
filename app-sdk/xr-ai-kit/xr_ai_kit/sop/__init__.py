# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard operating procedures: steps, a procedure that runs them in order, and a runner that runs it with a person.

The package, by layer::

    step.py         what a step is: the Step contract, the StepIO it touches, the roles it uses, what an evaluation
                    records of it (shared by every layer below)
    steps/          the step types, one package each, with everything the type uses inside it
      watch/          WatchStep: the fast check (judge, verify) and its optional per-step Escalator (escalate)
      read/           ReadStep: read the camera into a schema, derive a result (identify)
      wait/           WaitStep: a countdown (timer)
    procedure/      writing a procedure: its words in YAML (Wording) and the steps built from them; the procedure
                    itself is the app's ``async def procedure(run)``: which steps, in what order, what passes between
    runner/         SopRunner (a component: one run per participant, controls as a router destination, the run's
                    summary for the conversation), Run (what the procedure receives and what a step touches), the
                    controls' default wording
    eval/           one step at a time on a recorded scene and a scripted person, scored by the step type's own
                    ``score``; imported only to evaluate
    capture/        record a procedure being done (frames, captions, a packet) to write or evaluate one from

Everything a procedure needs is importable from here (``from xr_ai_kit.sop import WatchStep, Wording, SopRunner``);
the subpackages hold the parts underneath.

A procedure::

    WORDS = Wording.load("steps.yaml")
    fill = WORDS.watch("fill_water", roles).escalating(WORDS.escalator("fill_water", reasoning))
    steep = WORDS.wait("steep")

    async def make_tea(run: Run) -> None:
        if await run.step(fill):
            await run.say(WORDS.line("water_seen"))
        await run.step(steep.lasting(120))
"""

from __future__ import annotations

from .procedure import StepWords, Wording
from .runner import CONTROLS, MESSAGES, Run, SopRunner, StepRecord
from .step import Frames, Line, Roles, Step, StepIO, StepTrace
from .steps.read import Identification, ReadStep, identify
from .steps.wait import TimerState, WaitStep, timer
from .steps.watch import (
    LABELS,
    Check,
    Criteria,
    Escalation,
    EscalationSettings,
    Escalator,
    Facts,
    Judgment,
    Label,
    StepContext,
    Verification,
    Watched,
    WatchSettings,
    WatchStep,
    escalation_for,
    judge,
    stuck,
    verify,
)

__all__ = [
    "CONTROLS",
    "Check",
    "Criteria",
    "Escalation",
    "EscalationSettings",
    "Escalator",
    "Facts",
    "Frames",
    "Identification",
    "Judgment",
    "LABELS",
    "Label",
    "Line",
    "MESSAGES",
    "ReadStep",
    "Roles",
    "Run",
    "SopRunner",
    "Step",
    "StepContext",
    "StepIO",
    "StepRecord",
    "StepTrace",
    "StepWords",
    "TimerState",
    "Verification",
    "WaitStep",
    "WatchSettings",
    "WatchStep",
    "Watched",
    "Wording",
    "escalation_for",
    "identify",
    "judge",
    "stuck",
    "timer",
    "verify",
]
