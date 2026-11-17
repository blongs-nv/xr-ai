# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The watch step: the camera confirms something, and the person helps when the camera alone cannot.

==================  ==========================================================================================
``watch_step``      :class:`WatchStep` and :class:`WatchSettings`: what runs
``judge``           one frame against a :class:`Check` (observe, then decide a label); :class:`StepContext`
                    holds what is known beyond the camera
``verify``          the fast check: :func:`judge` in a loop until enough distinct frames confirm it
``escalator``       an :class:`Escalator`, attached per step, that asks the person or tells them what to show
                    when the check is stuck
``watch_scoring``   how an evaluation case of a watch step is scored (evaluation only; not imported here)
==================  ==========================================================================================
"""

from __future__ import annotations

from .escalator import STUCK, Escalation, EscalationSettings, Escalator, Watched, escalation_for, stuck
from .judge import LABELS, Check, Criteria, Facts, Judgment, Label, StepContext, judge
from .verify import Verification, verify
from .watch_step import WatchSettings, WatchStep

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
    "WatchSettings",
    "WatchStep",
    "Watched",
    "escalation_for",
    "judge",
    "stuck",
    "verify",
]
