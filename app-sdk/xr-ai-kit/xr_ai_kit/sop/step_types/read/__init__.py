# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The read step: read something from the camera into a schema, then derive a result from it.

``read_step`` is :class:`ReadStep` (what runs); ``identify`` is the reading and deriving underneath
(:func:`identify`); ``read_scoring`` is how an evaluation case of it is scored (evaluation only; not imported here).
"""

from __future__ import annotations

from .identify import Identification, identify
from .read_step import ReadStep, ReadWords

__all__ = [
    "Identification",
    "ReadStep",
    "ReadWords",
    "identify",
]
