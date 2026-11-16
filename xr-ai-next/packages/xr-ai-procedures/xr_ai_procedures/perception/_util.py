# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared by the perception plugs."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


async def call(fn: Callable[[Any], Any], arg: Any) -> Any:
    """Call a sync or async plug."""
    result = fn(arg)
    return await result if inspect.isawaitable(result) else result
