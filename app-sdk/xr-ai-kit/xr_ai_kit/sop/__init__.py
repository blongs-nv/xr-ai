# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard operating procedures: steps, a procedure that runs them in order, and a runner that runs it with a person.

The package::

    steps/            what every step shares, whatever its type
      step_api.py         Step, StepIO, Frames, and how a step type plugs in (models, define, closing, scoring)
      step_definitions.py StepDefinitions: a procedure's steps written in YAML (does, say, done; and sections,
                          each read by its owner)
    step_types/       the kit's step types, one package each; an app's own follow the same shape
      watch/              WatchStep: the fast camera check (judge, verify); supports escalation
      read/               ReadStep: read the camera into a schema, derive a result (identify)
      wait/               WaitStep: a countdown (timer)
                          (each also has a *_scoring.py module, used only by the evaluation)
    escalation/       an agent of its own beside a step: it reads the step's outbound context, gives it inbound
                      context, and talks to the person itself (Link, Escalator, Escalated); optional
    runner/           SopRunner (one run per participant, controls, the run's summary), Run (what the procedure
                      receives), the controls' default wording
    evaluation/       one step at a time on a recorded scene and a scripted person; imported only to evaluate
    capture/          record a procedure being done, to write or evaluate one from

The procedure itself is the app's own ``async def procedure(run)``: which steps, in what order, what passes
between them. Everything it needs is importable from here.

A procedure::

    STEPS = StepDefinitions.load("steps.yaml")
    fill = Escalated(
        WatchStep.define(STEPS["fill_water"], observe=vlm, decide=classifier),
        ReportEscalator.define(STEPS["fill_water"], reasoning),
    )
    steep = WaitStep.define(STEPS["steep"])

    async def make_tea(run: Run) -> None:
        await run.step(fill)                  # says its say, then its done
        await run.step(steep.lasting(120))

Each step names the models it uses as its own fields, so steps may use different models, and a step type of the
app's own (any object with ``id``, ``say`` and ``run(io)``) runs and is evaluated like the kit's.
"""

from __future__ import annotations

from .escalation import (
    STUCK,
    Escalatable,
    Escalated,
    Escalation,
    EscalationIO,
    EscalationSettings,
    EscalationWords,
    Escalator,
    Inbound,
    Link,
    Outbound,
    Report,
    ReportEscalator,
    escalation_for,
    stuck,
)
from .runner import CONTROLS, MESSAGES, Run, SopRunner, StepRecord
from .step_types.read import Identification, ReadStep, ReadWords, identify
from .step_types.wait import TimerState, WaitStep, timer
from .step_types.watch import (
    LABELS,
    Check,
    Criteria,
    Facts,
    Judgment,
    Label,
    StepContext,
    Verification,
    WatchSettings,
    WatchStep,
    WatchWords,
    judge,
    verify,
)
from .steps import Frames, Step, StepDefinition, StepDefinitions, StepIO

__all__ = [
    "CONTROLS",
    "Check",
    "Criteria",
    "Escalatable",
    "Escalated",
    "Escalation",
    "EscalationIO",
    "EscalationSettings",
    "Escalator",
    "EscalationWords",
    "Facts",
    "Frames",
    "Identification",
    "Inbound",
    "Judgment",
    "LABELS",
    "Label",
    "Link",
    "MESSAGES",
    "Outbound",
    "ReadStep",
    "ReadWords",
    "Report",
    "ReportEscalator",
    "Run",
    "STUCK",
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
    "WatchWords",
    "escalation_for",
    "identify",
    "judge",
    "stuck",
    "timer",
    "verify",
]
