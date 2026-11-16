# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Procedures (SOPs): capture a demonstrated procedure, then guide someone through an approved one.

A kit on top of the ``xr_ai_app`` SDK for apps built around standard operating procedures. The SDK gives the camera
stream (``p.frames``) and the perception role; how a step watches the camera over time (``Verify``, ``read_check``,
``Watching``, ``Done``/``Needs``/``Stuck``) is this kit's ``xr_ai_procedures.perception``.

=========================  ===========================================================================================
The procedure              Sop, StepSpec (the words and the order, YAML); StepDef (what a step does: a builder that
                           returns any ``Runnable``, and the facts the step owns); Run, Facts (one run's state);
                           GuidedProcedure (the supervisor, the controls, the summary channel)
Step kinds                 a camera check (``perception.read_check``), a read-and-look-up
                           (``perception.VisionRetrieval``), Timer, or anything of your own with ``run(report)``
Watching the camera        ``xr_ai_procedures.perception``: the outcomes, Eyes, Watching, Confirmation, Verify,
                           read_check, the plugs, VisionRetrieval (its own ``__all__``)
Capture                    ProcedureCapture, Caption, CaptureSession, caption_prompt: a narrated demonstration as an
                           evidence packet, the input for writing a guide
Events and control         procedure.on(kind, handler) with ProcedureEvent, EventKind; procedure.start/stop/next/back
                           (no user turn needed)
Testing                    the devkit targets ``step:`` (one step on a video window: step_unit) and ``procedure:``
                           (the whole procedure on a recording, against annotated windows: procedure_unit,
                           ProcedureRun, StepTiming)
=========================  ===========================================================================================

Wiring::

    procedure = app.add(GuidedProcedure(sop=Sop.load("sop.yaml"), steps=[StepDef("bed_clear", build), ...],
                                        output=voice, decide=decide, reason=reasoning))
    Conversation(..., context=[voice.dialogue, procedure.channel])
    chat = OneShotDestination("conversation", "Questions, also about the procedure.", conversation.answer)
    Router(..., destinations=[chat, procedure.control], default=chat)
"""

from __future__ import annotations

from .capture import DEFAULT_CAPTURE_MESSAGES, Caption, CaptureSession, ProcedureCapture, caption_prompt
from .events import EventKind, ProcedureEvent
from .exercise import (
    ProcedureRun,
    StepTiming,
    procedure_unit,
    procedures_of,
    resolve_procedure,
    resolve_step,
    step_unit,
)
from .guide import GuidedProcedure
from .run import Facts, Run
from .spec import DEFAULT_ASKS, DEFAULT_MESSAGES, Sop, StepSpec
from .step import StepDef
from .timer import Timer

__all__ = [
    "DEFAULT_ASKS",
    "DEFAULT_CAPTURE_MESSAGES",
    "DEFAULT_MESSAGES",
    "Caption",
    "CaptureSession",
    "EventKind",
    "Facts",
    "GuidedProcedure",
    "ProcedureCapture",
    "ProcedureEvent",
    "ProcedureRun",
    "Run",
    "Sop",
    "StepDef",
    "StepSpec",
    "StepTiming",
    "Timer",
    "caption_prompt",
    "procedure_unit",
    "procedures_of",
    "resolve_procedure",
    "resolve_step",
    "step_unit",
]
