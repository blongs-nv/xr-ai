# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The wait step: a countdown; no camera, no models.

``wait_step`` is :class:`WaitStep` (what runs); ``timer`` is the countdown underneath (:func:`timer`);
``wait_scoring`` is how an evaluation case of it is scored (evaluation only; not imported here).
"""

from __future__ import annotations

from .timer import TimerState, timer
from .wait_step import WaitInputs, WaitStep

__all__ = [
    "TimerState",
    "WaitInputs",
    "WaitStep",
    "timer",
]
