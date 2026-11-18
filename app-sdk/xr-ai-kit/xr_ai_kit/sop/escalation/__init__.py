# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Escalation: an agent of its own beside a step, helping the person when the step cannot decide.

=========================  ===========================================================================================
``escalation_channel``     :class:`EscalationChannel`, the only thing a step and its escalator share: a
                           :class:`StepSnapshot` of the step for the escalator, and :class:`AddedContext` from the
                           escalator for the step; :class:`SupportsEscalation`, what a step type implements to
                           support it; :class:`Escalator`, what any escalator implements; :class:`EscalatorEnvironment`,
                           how an escalator reaches the person and the camera
``reasoning_escalator``    :class:`ReasoningEscalator`, the kit's :class:`Escalator`: when the step is stuck, reason
                           over its evidence and ask the person or tell them what to do; its prompt is per step
``step_with_escalator``    :class:`StepWithEscalator`: a step with an escalator beside it, itself a step
``escalation_scoring``     what an evaluation case may expect of an escalator (evaluation only; not imported here)
=========================  ===========================================================================================

Optional: the core step contract does not mention it, and a step type supports it only by implementing
:class:`SupportsEscalation`. Nothing here imports a step type.
"""

from __future__ import annotations

from .escalation_channel import (
    ESCALATOR_SPEAKING,
    AddedContext,
    EscalationChannel,
    Escalator,
    EscalatorEnvironment,
    Evidence,
    StepSnapshot,
    SupportsEscalation,
    speaker,
)
from .reasoning_escalator import (
    INSTRUCTIONS,
    PROMPT,
    STUCK_RULE,
    EscalationMessage,
    EscalationSettings,
    EscalationWords,
    ReasoningEscalator,
    decide_escalation,
    is_stuck,
)
from .step_with_escalator import StepWithEscalator

__all__ = [
    "ESCALATOR_SPEAKING",
    "INSTRUCTIONS",
    "PROMPT",
    "STUCK_RULE",
    "AddedContext",
    "EscalationChannel",
    "EscalationMessage",
    "EscalationSettings",
    "EscalationWords",
    "Escalator",
    "EscalatorEnvironment",
    "Evidence",
    "ReasoningEscalator",
    "StepSnapshot",
    "StepWithEscalator",
    "SupportsEscalation",
    "decide_escalation",
    "is_stuck",
    "speaker",
]
