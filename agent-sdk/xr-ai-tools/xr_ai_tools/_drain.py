# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wait for work that Relay runs outside the awaiting task to finish."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager

_LOGGER = logging.getLogger(__name__)
DRAIN_TIMEOUT_S = 5.0


class ActiveCalls:
    """Track handler and model calls one request started.

    Relay runs a tool handler or model call outside the task that awaits it,
    so cancelling the awaiting task can return while that work is still open
    (NVIDIA/xr-ai#697). Callers track the work and await :meth:`drained` when
    they are cancelled, so cancellation returns only after the work stopped.
    """

    def __init__(self) -> None:
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @contextmanager
    def track(self) -> Iterator[None]:
        self._active += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._active -= 1
            if self._active == 0:
                self._idle.set()

    async def drained(self) -> None:
        """Wait, bounded, until every tracked call has finished."""

        try:
            await asyncio.wait_for(self._idle.wait(), DRAIN_TIMEOUT_S)
        except TimeoutError:
            _LOGGER.warning("call still running %.1fs after its request ended", DRAIN_TIMEOUT_S)
