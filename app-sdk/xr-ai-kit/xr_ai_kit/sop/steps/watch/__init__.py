# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The watch step: the camera confirms something, and the person helps when the camera alone cannot.

=============  ===============================================================================================
``step``       :class:`WatchStep` (the step: its settings, ``run``, and how a case of it is scored)
``judge``      one frame against a :class:`Check` (observe, then decide a label); :class:`StepContext` holds
               what is known beyond the camera
``verify``     the fast check: :func:`judge` in a loop until enough distinct frames confirm it
``escalate``   the step's side escalation: an :class:`Escalator` attached per step, asking the person or
               telling them what to show when the check is stuck
=============  ===============================================================================================
"""

from __future__ import annotations

from .escalate import STUCK, Escalation, EscalationSettings, Escalator, Watched, escalation_for, stuck
from .judge import LABELS, Check, Criteria, Facts, Judgment, Label, StepContext, judge
from .step import WatchExpect, WatchSettings, WatchStep, Window
from .verify import Verification, verify

__all__ = [
    "Check",
    "Criteria",
    "Escalation",
    "EscalationSettings",
    "Escalator",
    "Facts",
    "Judgment",
    "LABELS",
    "Label",
    "STUCK",
    "StepContext",
    "Verification",
    "WatchExpect",
    "WatchSettings",
    "WatchStep",
    "Watched",
    "Window",
    "escalation_for",
    "judge",
    "stuck",
    "verify",
]
