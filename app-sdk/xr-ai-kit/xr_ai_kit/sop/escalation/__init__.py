# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Escalation: an agent of its own beside a step, helping the person when the step cannot decide.

=========================  ===========================================================================================
``link``                   :class:`Link`, the only thing a step and its escalator share: the step's outbound
                           context (:class:`Outbound`) and the escalator's inbound context (:class:`Inbound`);
                           :class:`Escalatable`, what a step type implements to support it; :class:`EscalationIO`,
                           how the escalator reaches the person
``escalator``              :class:`Escalator`: when the step is stuck, ask the person or tell them what to do
``escalated``              :class:`Escalated`: a step with an escalator beside it, itself a step
``escalation_scoring``     what an evaluation case may expect of an escalator (evaluation only; not imported here)
=========================  ===========================================================================================

Optional: the core step contract does not mention it, and a step type supports it only by implementing
:class:`Escalatable`. Nothing here imports a step type.
"""

from __future__ import annotations

from .escalated import Escalated
from .escalator import STUCK, Escalation, EscalationSettings, Escalator, escalation_for, stuck
from .link import Escalatable, EscalationIO, Inbound, Link, Outbound, Report

__all__ = [
    "STUCK",
    "Escalatable",
    "Escalated",
    "Escalation",
    "EscalationIO",
    "EscalationSettings",
    "Escalator",
    "Inbound",
    "Link",
    "Outbound",
    "Report",
    "escalation_for",
    "stuck",
]
