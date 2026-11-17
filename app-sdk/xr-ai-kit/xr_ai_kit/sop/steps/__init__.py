# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every step shares, whatever its type; the step types are the sibling packages ``watch``, ``read``, ``wait``.

=======================  ====================================================================================
``step_api``             what every step is and touches: :class:`Step`, :class:`StepIO`, :class:`Frames`, and
                         how a step type plugs in (its models, ``define``, ``scoring``)
``step_definitions``     :class:`StepDefinitions`: a procedure's steps written in YAML; it reads only what
                         every step has (``does``, ``say``) and leaves the rest to each step's type
=======================  ====================================================================================

Nothing here imports a step type.
"""

from __future__ import annotations

from .step_api import Frames, Step, StepIO
from .step_definitions import StepDefinition, StepDefinitions

__all__ = [
    "Frames",
    "Step",
    "StepDefinition",
    "StepDefinitions",
    "StepIO",
]
