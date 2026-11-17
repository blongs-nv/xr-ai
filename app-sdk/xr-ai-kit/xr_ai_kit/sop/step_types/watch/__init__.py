# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The watch step: the camera confirms something. It supports escalation (``...escalation``).

==================  ==========================================================================================
``watch_step``      :class:`WatchStep` and :class:`WatchSettings`: what runs
``judge``           one frame against a :class:`Check` (observe, then decide a label); :class:`StepContext`
                    holds what is known beyond the camera
``verify``          the fast check: :func:`judge` in a loop until enough distinct frames confirm it
``watch_scoring``   how an evaluation case of a watch step is scored (evaluation only; not imported here)
==================  ==========================================================================================
"""

from __future__ import annotations

from .judge import LABELS, Check, Criteria, Facts, Judgment, Label, StepContext, judge
from .verify import Verification, verify
from .watch_step import WatchSettings, WatchStep, WatchWords, outbound

__all__ = [
    "Check",
    "Criteria",
    "Facts",
    "Judgment",
    "LABELS",
    "Label",
    "StepContext",
    "Verification",
    "WatchSettings",
    "WatchStep",
    "WatchWords",
    "judge",
    "outbound",
    "verify",
]
