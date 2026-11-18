# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The read step: read something from the camera into a schema, until what has been read is enough.

``read_step`` is :class:`ReadStep` (what runs); ``read_until_enough`` is the reading underneath
(:func:`read_until_enough`, merging what each frame shows); ``read_scoring`` is how an evaluation case of it is
scored (evaluation only; not imported here).
"""

from __future__ import annotations

from .read_step import ReadSection, ReadSettings, ReadStep
from .read_until_enough import ReadState, merge, read_until_enough

__all__ = [
    "ReadSection",
    "ReadSettings",
    "ReadState",
    "ReadStep",
    "merge",
    "read_until_enough",
]
