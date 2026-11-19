# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The watch step: the camera confirms something. It supports escalation (``...escalation``).

==================  ==========================================================================================
``watch_step``      :class:`WatchStep` and :class:`WatchWords`: what runs
``judge``           one frame against a :class:`Check` with :class:`WatchModels` (perceive, then decide a label);
                    :class:`StructuredObservation` reads a frame into a schema, optionally judged in code
                    (:class:`CodeJudge`); :class:`Knowledge` and :class:`StepKnowledge`: what a step knows beyond
                    the camera
``verify``          the fast check: :func:`judge` in a loop until enough distinct frames confirm it
                    (:class:`WatchSettings`)
``watch_scoring``   how an evaluation case of a watch step is scored (evaluation only; not imported here)
==================  ==========================================================================================

Patterns, what works and what does not, smaller models and speed: README.md, next to this file.
"""

from __future__ import annotations

from .judge import (
    LABELS,
    Check,
    CodeJudge,
    Criteria,
    Judgment,
    Knowledge,
    Label,
    StepKnowledge,
    StructuredObservation,
    WatchModels,
    judge,
)
from .verify import Verification, WatchSettings, verify
from .watch_step import WatchStep, WatchWords

__all__ = [
    "Check",
    "CodeJudge",
    "Criteria",
    "Judgment",
    "Knowledge",
    "LABELS",
    "Label",
    "StepKnowledge",
    "StructuredObservation",
    "Verification",
    "WatchModels",
    "WatchWords",
    "WatchSettings",
    "WatchStep",
    "judge",
    "verify",
]
