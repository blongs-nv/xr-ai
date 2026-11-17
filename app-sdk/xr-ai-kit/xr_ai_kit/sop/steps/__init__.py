# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The step types, one package each, with everything the type uses inside it.

=========  ==========================================================================================
watch      the camera confirms something (the fast check), the person helps when it cannot (an optional
           escalator, attached per step: asks for what is missing, or tells them what to bring into view)
read       read something from the camera into a schema, then derive a result (a label, its settings)
wait       a countdown
=========  ==========================================================================================

Each type owns ``run(io)`` and how an evaluation case of it is scored (``Expect``, ``score``). What every step
shares (the :class:`~..step.Step` contract, :class:`~..step.StepIO`) is in :mod:`..step`.
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
