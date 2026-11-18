# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every step shares, whatever its type; the step types are the sibling packages ``watch``, ``read``, ``wait``.

=======================  ====================================================================================
``step_api``             what every step is and touches: :class:`Step`, :class:`StepEnvironment`,
                         :class:`Frames`, and how a step's words get its inputs (:func:`check_placeholders`,
                         :func:`fill_placeholders`)
``step_words``           :class:`StepWords`, what every step says, which each step type's words extend, and
                         :func:`load_words`, YAML into such a schema
=======================  ====================================================================================

Nothing here imports a step type.
"""

from __future__ import annotations

from .step_api import Frames, Step, StepEnvironment, check_inputs, check_placeholders, fill_placeholders
from .step_words import StepWords, load_words

__all__ = [
    "Frames",
    "Step",
    "StepWords",
    "StepEnvironment",
    "check_inputs",
    "check_placeholders",
    "fill_placeholders",
    "load_words",
]
