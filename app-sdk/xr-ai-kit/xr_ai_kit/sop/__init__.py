# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard operating procedures: steps, a procedure that runs them in order, and a runner that runs it with a person.

The package::

    steps/            everything about steps
      step_api.py         what every step is and touches (Step, StepIO, Frames, Roles)
      step_definitions.py StepDefinitions: a procedure's steps written in YAML, and the steps built from them
      watch/              WatchStep: the fast camera check (judge, verify) and its per-step Escalator
      read/               ReadStep: read the camera into a schema, derive a result (identify)
      wait/               WaitStep: a countdown (timer)
                          (each type also has a *_scoring.py module, used only by the evaluation)
    runner/           SopRunner (one run per participant, controls, the run's summary), Run (what the procedure
                      receives), the controls' default wording
    evaluation/       one step at a time on a recorded scene and a scripted person; imported only to evaluate
    capture/          record a procedure being done, to write or evaluate one from

The procedure itself is the app's own ``async def procedure(run)``: which steps, in what order, what passes
between them. Everything it needs is importable from here.

A procedure::

    STEPS = StepDefinitions.load("steps.yaml")
    fill = STEPS.watch("fill_water", roles).escalating(STEPS.escalator("fill_water", reasoning))
    steep = STEPS.wait("steep")

    async def make_tea(run: Run) -> None:
        if await run.step(fill):
            await run.say(STEPS.line("water_seen"))
        await run.step(steep.lasting(120))
"""

from __future__ import annotations

from .runner import CONTROLS, MESSAGES, Run, SopRunner, StepRecord
from .steps import Frames, Roles, Step, StepDefinition, StepDefinitions, StepIO
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
    "MESSAGES",
    "ReadStep",
    "Roles",
    "Run",
    "SopRunner",
    "Step",
    "StepContext",
    "StepDefinition",
    "StepDefinitions",
    "StepIO",
    "StepRecord",
    "TimerState",
    "Verification",
    "WaitStep",
    "WatchSettings",
    "WatchStep",
    "Watched",
    "escalation_for",
    "identify",
    "judge",
    "stuck",
    "timer",
    "verify",
]
