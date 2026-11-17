# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The step types. Each owns ``run(io)`` and how an evaluation case of it is scored (``Expect``, ``score``).

=========  ==========================================================================================
WatchStep  the camera confirms something (the fast check), the person helps when it cannot (an optional
           escalator, attached per step: asks for what is missing, or tells them what to bring into view)
ReadStep   read something from the camera into a schema, then derive a result (a label, its settings)
WaitStep   a countdown
=========  ==========================================================================================
"""

from __future__ import annotations

from .read import ReadExpect, ReadStep
from .wait import WaitExpect, WaitStep
from .watch import WatchExpect, WatchSettings, WatchStep, Window

__all__ = [
    "ReadExpect",
    "ReadStep",
    "WaitExpect",
    "WaitStep",
    "WatchExpect",
    "WatchSettings",
    "WatchStep",
    "Window",
]
