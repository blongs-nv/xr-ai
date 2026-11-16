# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The XR-AI kit: common app patterns built only from ``xr_ai_next``'s public names.

The core (``xr_ai_next``) holds the primitives that need the runtime, the hub or the model binding; everything here is
composed from them, however common. Modules: ``xr_ai_kit.conversation`` (answers with a tool loop),
``.routing`` (also ``choose``, the pick the Router is built on, and the devkit's ``destination:`` exercise target),
``.delegate`` (also the job back ends ``FunctionBackend`` and ``A2ABackend``, extra ``xr-ai-kit[a2a]``). The
procedures kit (``xr-ai-procedures``) is a separate package for now (``xr_ai_kit/procedures/README.md``).

======================  =================================================================
Answers                 Conversation (multi-turn answers over listed context, with a tool
                        loop over ``@tool`` functions)
Routing                 Router, OneShotDestination (a plain handler), StartStopDestination
                        (a mode started and stopped by asking), RouterDestination (any
                        custom destination: ``admit`` and ``focused`` hooks),
                        DestinationResult, Admission, Ack (routing by hand:
                        ``role.decide``)
Delegation              Delegate, Job
======================  =================================================================
"""

from __future__ import annotations

from .conversation import Conversation
from .delegate import Delegate, Job
from .routing import (
    Ack,
    Admission,
    DestinationResult,
    OneShotDestination,
    Router,
    RouterDestination,
    StartStopDestination,
)

__all__ = [
    "Ack",
    "Admission",
    "Conversation",
    "Delegate",
    "DestinationResult",
    "Job",
    "OneShotDestination",
    "Router",
    "RouterDestination",
    "StartStopDestination",
]
