# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Escalation: an agent of its own beside a step, helping the person when the step cannot decide.

=========================  ===========================================================================================
``link``                   :class:`Link`, the only thing a step and its escalator share: the step's outbound
                           context (:class:`Outbound`) and the escalator's inbound context (:class:`Inbound`);
                           :class:`Escalatable`, what a step type implements to support it; :class:`Escalator`,
                           what any escalator implements; :class:`EscalationIO`, how it reaches the person and
                           the camera
``escalator``              :class:`ReportEscalator`, the kit's :class:`Escalator`: when the step is stuck, reason
                           over its reports and ask the person or tell them what to do; its prompt is per step
``escalated``              :class:`Escalated`: a step with an escalator beside it, itself a step
``escalation_scoring``     what an evaluation case may expect of an escalator (evaluation only; not imported here)
=========================  ===========================================================================================

Optional: the core step contract does not mention it, and a step type supports it only by implementing
:class:`Escalatable`. Nothing here imports a step type.
"""

from __future__ import annotations

from .escalated import Escalated
from .escalator import (
    INSTRUCTIONS,
    PROMPT,
    STUCK,
    Escalation,
    EscalationSettings,
    EscalationWords,
    ReportEscalator,
    escalation_for,
    stuck,
)
from .link import Escalatable, EscalationIO, Escalator, Inbound, Link, Outbound, Report

__all__ = [
    "INSTRUCTIONS",
    "PROMPT",
    "STUCK",
    "Escalatable",
    "Escalated",
    "Escalation",
    "EscalationIO",
    "EscalationSettings",
    "EscalationWords",
    "Escalator",
    "Inbound",
    "Link",
    "Outbound",
    "Report",
    "ReportEscalator",
    "escalation_for",
    "stuck",
]
