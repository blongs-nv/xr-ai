# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every step shares, whatever its type; the step types are the sibling packages ``watch``, ``read``, ``wait``.

=======================  ====================================================================================
``step_api``             what every step is and touches: :class:`Step`, :class:`StepIO`, :class:`Frames`,
                         :class:`Roles` (the step types, the runner and the evaluation use it)
``step_definitions``     :class:`StepDefinitions`: a procedure's steps written in YAML (what is said, the
                         camera's question and criteria, escalation guidance), and the steps built from them
=======================  ====================================================================================

Importing this package loads only ``step_api``: ``step_definitions`` builds the step types, which import
``step_api``, so it is imported by name (``xr_ai_kit.sop`` re-exports it).
"""

from __future__ import annotations

from .step_api import Frames, Roles, Step, StepIO

__all__ = [
    "Frames",
    "Roles",
    "Step",
    "StepIO",
]
