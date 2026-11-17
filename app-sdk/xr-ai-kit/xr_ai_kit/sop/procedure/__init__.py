# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writing a procedure: its words in YAML (:class:`Wording`), and the steps built from them.

The procedure itself is the app's own ``async def procedure(run)``; this package holds what helps write one.
"""

from __future__ import annotations

from .wording import StepWords, Wording

__all__ = [
    "StepWords",
    "Wording",
]
