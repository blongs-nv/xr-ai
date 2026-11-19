# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard operating procedures: steps, a procedure that runs them in order, and a runner that runs it with a person.

The package::

    steps/            what every step shares, whatever its type
      step_api.py         Step, StepEnvironment, Frames: what a step is and what it may touch
      step_words.py       StepWords (what every step says; each step type's words extend it) and load_words
                          (YAML into a words schema)
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

    fill_water = StepWithEscalator(
        WatchStep("fill_water", load_words(HERE / "fill_water.yaml", WatchWords),
                  WatchModels(perceive=vision_model, decide=classifier)),
        ReasoningEscalator(load_words(HERE / "fill_water.escalation.yaml", EscalationWords), reasoning_model),
    )
    steep = WaitStep("steep", load_words(HERE / "steep.yaml", StepWords), inputs_schema=SteepInputs)

    async def make_tea(run: Run) -> None:
        await run.step(fill_water)                                # says its start_line, then its done_line
        await run.step(steep, SteepInputs(duration_s=120))

Each step is defined once, from its id, its words, the models it uses and its ``inputs_schema``, and each run of it
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
    EscalationSettings,
    EscalationWords,
    Escalator,
    EscalatorEnvironment,
    Evidence,
    ReasoningEscalator,
    StepSnapshot,
    StepWithEscalator,
    SupportsEscalation,
    decide_escalation,
    is_stuck,
    read_facts_from_conversation,
)
from .runner import CONTROLS, MESSAGES, Run, SopRunner, StepRecord
from .step_types.read import ReadSettings, ReadState, ReadStep, ReadWords, read_until_enough
from .step_types.transform import TransformSettings, TransformState, TransformStep, TransformWords
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
    WatchSettings,
    WatchStep,
    WatchWords,
    judge,
    verify,
)
from .steps import (
    Conversation,
    FixedConversation,
    Frames,
    Step,
    StepEnvironment,
    StepWords,
    check_inputs,
    check_placeholders,
    fill_placeholders,
    load_words,
)

__all__ = [
    "CONTROLS",
    "AddedContext",
    "Check",
    "Conversation",
    "Criteria",
    "EscalationChannel",
    "EscalationMessage",
    "EscalationSettings",
    "EscalationWords",
    "Escalator",
    "EscalatorEnvironment",
    "Evidence",
    "Facts",
    "FixedConversation",
    "Frames",
    "Judgment",
    "LABELS",
    "Label",
    "MESSAGES",
    "ReadSettings",
    "ReadState",
    "ReadStep",
    "ReadWords",
    "ReasoningEscalator",
    "Run",
    "STUCK_RULE",
    "SopRunner",
    "Step",
    "StepContext",
    "StepWords",
    "StepEnvironment",
    "StepRecord",
    "StepSnapshot",
    "StepWithEscalator",
    "StructuredObservation",
    "SupportsEscalation",
    "TimerState",
    "TransformSettings",
    "TransformState",
    "TransformStep",
    "TransformWords",
    "Verification",
    "WaitInputs",
    "WaitStep",
    "WatchModels",
    "WatchWords",
    "WatchSettings",
    "WatchStep",
    "check_inputs",
    "check_placeholders",
    "decide_escalation",
    "fill_placeholders",
    "is_stuck",
    "read_facts_from_conversation",
    "judge",
    "load_words",
    "read_until_enough",
    "timer",
    "verify",
]
