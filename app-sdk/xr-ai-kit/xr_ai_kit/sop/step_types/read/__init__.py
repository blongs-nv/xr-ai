# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The read step: read something from the camera into a schema, until what has been read is enough.

``read_step`` is :class:`ReadStep` (what runs); ``read_until_enough`` is the reading underneath
(:func:`read_until_enough`, merging what each frame shows); ``read_scoring`` is how an evaluation case of it is
scored (evaluation only; not imported here).

Patterns, what works and what does not, smaller models and speed: README.md, next to this file.
"""

from __future__ import annotations

from .read_step import ReadStep, ReadWords
from .read_until_enough import ReadSettings, ReadState, read_until_enough

__all__ = [
    "ReadSettings",
    "ReadState",
    "ReadStep",
    "ReadWords",
    "read_until_enough",
]
