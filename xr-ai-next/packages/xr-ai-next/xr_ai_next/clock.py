# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time, the way app and library code should read it.

Use these instead of ``time.monotonic()``, ``time.time()`` and bare network calls, and the code is correct both live
and on the devkit's virtual-time loop (which skips idle waits so long tests run in moments):

- :func:`now`: seconds on the running event loop's clock, for elapsed time and deadlines (``time.monotonic()``
  live).
- :func:`wall_us`: wall-clock microseconds, for frame capture times and frame ages (a frame is ``wall_us() -
  frame.timestamp_us`` old).
- :func:`held`: wrap real external work your code starts itself (a network call that is not a role call), so a
  virtual clock waits for it for real instead of hitting its timeouts early. Role calls are held already.
  :func:`held_call` and :class:`HeldStream` do the same for an async function and an async iterator.
"""

from __future__ import annotations

import asyncio
import time

from ._base._virtual import HeldStream, held, held_call, wall_us

__all__ = ["HeldStream", "held", "held_call", "now", "wall_us"]


def now() -> float:
    """Seconds on the running event loop's clock (``time.monotonic()`` outside one, and live)."""
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        return time.monotonic()
