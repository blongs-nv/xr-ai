# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything about steps: what a step is, the step types, and building steps from a YAML file.

=======================  ====================================================================================
``step_api``             what every step is and touches: :class:`Step`, :class:`StepIO`, :class:`Frames`,
                         :class:`Roles` (the runner and the evaluation use it too)
``step_definitions``     :class:`StepDefinitions`: a procedure's steps written in YAML (what is said, the
                         camera's question and criteria, escalation guidance), and the steps built from them
``watch/``               the camera confirms something; an optional per-step escalator helps when it cannot
``read/``                read something from the camera into a schema, then derive a result
``wait/``                a countdown
=======================  ====================================================================================

Each step type is a package with everything it uses inside it, and its scoring for evaluation in a module of its
own (``watch_scoring``, ...), which the running code never imports.
"""

from __future__ import annotations

from .read import ReadStep
from .step_api import Frames, Roles, Step, StepIO
from .step_definitions import StepDefinition, StepDefinitions
from .wait import WaitStep
from .watch import WatchSettings, WatchStep

__all__ = [
    "Frames",
    "ReadStep",
    "Roles",
    "Step",
    "StepDefinition",
    "StepDefinitions",
    "StepIO",
    "WaitStep",
    "WatchSettings",
    "WatchStep",
]
