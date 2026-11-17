# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The read step: read something from the camera into a schema, then derive a result from it.

``step`` is :class:`ReadStep` (the step, and how a case of it is scored); ``identify`` is the reading and deriving
underneath (:func:`identify`).
"""

from __future__ import annotations

from .identify import Identification, identify
from .step import ReadExpect, ReadStep

__all__ = [
    "Identification",
    "ReadExpect",
    "ReadStep",
    "identify",
]
