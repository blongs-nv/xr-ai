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
      read/               ReadStep: read the camera into a schema until it is enough (read_until_enough)
      transform/          TransformStep: a reasoning model turns one schema into another; may ask the person
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
    fill_water = StepWithEscalator(
        WatchStep(STEPS["fill_water"], WatchModels(perceive=vision_model, decide=classifier)),
        ReasoningEscalator(STEPS["fill_water"], reasoning_model),
    )
    steep = WaitStep(STEPS["steep"], inputs_schema=SteepInputs)   # SteepInputs(WaitInputs): {minutes} for its words

    async def make_tea(run: Run) -> None:
        await run.step(fill_water)                                # says its start_line, then its done_line
        await run.step(steep, SteepInputs(duration_s=120))

Each step is defined once, from its definition, the models it uses and its ``inputs_schema``, and each run of it
gets its inputs: what earlier steps produced, passed on by the procedure. A step type of the app's own (any object
with ``id``, ``inputs_schema``, ``start_line(inputs)``, ``done_line(inputs, result)`` and ``run(environment,
inputs)``) runs and is evaluated like the kit's.
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
from .step_types.read import ReadSection, ReadSettings, ReadState, ReadStep, read_until_enough
from .step_types.transform import TransformSection, TransformSettings, TransformState, TransformStep
from .step_types.wait import TimerState, WaitInputs, WaitStep, timer
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
from .steps import (
    Frames,
    Step,
    StepDefinition,
    StepDefinitions,
    StepEnvironment,
    check_inputs,
    check_placeholders,
    fill_placeholders,
)

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
    "ReadSection",
    "ReadSettings",
    "ReadState",
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
    "TransformSection",
    "TransformSettings",
    "TransformState",
    "TransformStep",
    "Verification",
    "WaitInputs",
    "WaitStep",
    "WatchModels",
    "WatchSection",
    "WatchSettings",
    "WatchStep",
    "check_inputs",
    "check_placeholders",
    "decide_escalation",
    "fill_placeholders",
    "is_stuck",
    "judge",
    "read_until_enough",
    "timer",
    "verify",
]
