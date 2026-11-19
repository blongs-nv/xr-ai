# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every step shares, whatever its type; the step types are in ``..step_types`` (watch, read, transform, wait).

=======================  ====================================================================================
``step_api``             what every step is and touches: :class:`Step`, :class:`StepEnvironment`,
                         :class:`Frames`, how a step's words get its inputs (:func:`check_placeholders`,
                         :func:`fill_placeholders`), and who is speaking (:func:`speaker`)
``step_words``           :class:`StepWords`, what every step says, which each step type's words extend, and
                         :func:`load_words`, YAML into such a schema
``conversation``         :class:`ConversationHistory`, the conversation with the person as a step and its
                         escalator see it (``environment.conversation``), and :class:`FixedConversationHistory`
=======================  ====================================================================================

Nothing here imports a step type.
"""

from __future__ import annotations

from .conversation import ConversationHistory, FixedConversationHistory
from .step_api import (
    SPEAKER,
    Frames,
    Step,
    StepEnvironment,
    check_inputs,
    check_placeholders,
    fill_placeholders,
    speaker,
)
from .step_words import StepWords, load_words

__all__ = [
    "SPEAKER",
    "ConversationHistory",
    "FixedConversationHistory",
    "Frames",
    "Step",
    "StepWords",
    "StepEnvironment",
    "speaker",
    "check_inputs",
    "check_placeholders",
    "fill_placeholders",
    "load_words",
]
