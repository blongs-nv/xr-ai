# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The read step: read something from the camera into a schema, then derive a result from it.

``read_step`` is :class:`ReadStep` (what runs); ``read_and_derive`` is the reading and deriving underneath
(:func:`read_and_derive`); ``read_scoring`` is how an evaluation case of it is scored (evaluation only; not imported
here).
"""

from __future__ import annotations

from .read_and_derive import ReadResult, read_and_derive
from .read_step import ReadModels, ReadSection, ReadSettings, ReadStep

__all__ = [
    "ReadModels",
    "ReadResult",
    "ReadSection",
    "ReadSettings",
    "ReadStep",
    "read_and_derive",
]
