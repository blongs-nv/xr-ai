# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard operating procedures: steps, a procedure that runs them in order, and a runner that runs it with a person.

The package::

    steps/            what every step shares, whatever its type
      step_api.py         Step, StepEnvironment, Frames: what a step is and what it may touch
      step_definitions.py StepDefinitions: a procedure's steps written in YAML (does, start_line, done_line; and
                          sections, each read by its owner)
    step_types/       the kit's step types, one package each; an app's own follow the same shape
      watch/              WatchStep: the fast camera check (judge, verify); supports escalation
      read/               ReadStep: read the camera into a schema, derive a result (read_and_derive)
      wait/               WaitStep: a countdown (timer)
                          (each also has a *_scoring.py module, used only by the evaluation)
    escalation/       an agent of its own beside a step (StepWithEscalator): through an EscalationChannel it reads
                      a snapshot of the step and adds context to it, and it talks to the person itself; optional
    runner/           SopRunner (one run per participant, controls, the run's summary), Run (what the procedure
                      receives), the controls' default wording
    evaluation/       one step at a time on a recorded scene and a scripted person; imported only to evaluate
    capture/          record a procedure being done, to write or evaluate one from

The procedure itself is the app's own ``async def procedure(run)``: which steps, in what order, what passes
between them. Everything it needs is importable from here.

A procedure::

    STEPS = StepDefinitions.load("steps.yaml")
    CAMERA = WatchModels(perceive=vision_model, decide=classifier)
    fill_water = StepWithEscalator(
        WatchStep(STEPS["fill_water"], CAMERA),
        ReasoningEscalator(STEPS["fill_water"], reasoning_model),
    )

    async def make_tea(run: Run) -> None:
        await run.step(fill_water)            # says its start_line, then its done_line
        await run.step(WaitStep(STEPS["steep"], 120, values={"minutes": 2}))

Each step is built in one call: its definition, the models it uses, and ``values`` for the placeholders in its
words. A step type of the app's own (any object with ``id``, ``start_line``, ``done_line(result)`` and
``run(environment)``) runs and is evaluated like the kit's.
"""

from __future__ import annotations

from .escalation import (
    STUCK_RULE,
    AddedContext,
    EscalationChannel,
    EscalationMessage,
    EscalationSection,
    EscalationSettings,
    Escalator,
    EscalatorEnvironment,
    Evidence,
    ReasoningEscalator,
    StepSnapshot,
    StepWithEscalator,
    SupportsEscalation,
    decide_escalation,
    is_stuck,
)
from .runner import CONTROLS, MESSAGES, Run, SopRunner, StepRecord
from .step_types.read import ReadModels, ReadResult, ReadSection, ReadSettings, ReadStep, read_and_derive
from .step_types.wait import TimerState, WaitStep, timer
from .step_types.watch import (
    LABELS,
    Check,
    Criteria,
    Facts,
    Judgment,
    Label,
    StepContext,
    StructuredObservation,
    Verification,
    WatchModels,
    WatchSection,
    WatchSettings,
    WatchStep,
    judge,
    verify,
)
from .steps import Frames, Step, StepDefinition, StepDefinitions, StepEnvironment, fill_placeholders

__all__ = [
    "CONTROLS",
    "AddedContext",
    "Check",
    "Criteria",
    "EscalationChannel",
    "EscalationMessage",
    "EscalationSection",
    "EscalationSettings",
    "Escalator",
    "EscalatorEnvironment",
    "Evidence",
    "Facts",
    "Frames",
    "Judgment",
    "LABELS",
    "Label",
    "MESSAGES",
    "ReadModels",
    "ReadResult",
    "ReadSection",
    "ReadSettings",
    "ReadStep",
    "ReasoningEscalator",
    "Run",
    "STUCK_RULE",
    "SopRunner",
    "Step",
    "StepContext",
    "StepDefinition",
    "StepDefinitions",
    "StepEnvironment",
    "StepRecord",
    "StepSnapshot",
    "StepWithEscalator",
    "StructuredObservation",
    "SupportsEscalation",
    "TimerState",
    "Verification",
    "WaitStep",
    "WatchModels",
    "WatchSection",
    "WatchSettings",
    "WatchStep",
    "decide_escalation",
    "fill_placeholders",
    "is_stuck",
    "judge",
    "read_and_derive",
    "timer",
    "verify",
]
