# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Confirmation``: is it really so? k of the last n usable verdicts said yes."""

from __future__ import annotations

from collections import deque

__all__ = ["Confirmation"]


class Confirmation:
    """Confirms a condition when ``required`` of the last ``window`` usable verdicts are ``"yes"``.

    ``add(verdict)`` takes ``"yes"`` or ``"no"`` into the window and ignores anything else (an ``"unknown"`` look
    is not evidence either way); it returns whether the condition is confirmed. ``required == window`` means that
    many yes in a row. A monitor keeps one per condition and calls ``reset()`` after acting on it.
    """

    def __init__(self, required: int, window: int) -> None:
        if not 1 <= required <= window:
            raise ValueError("required must be at least 1 and fit in the window")
        self.required = required
        self._window: deque[str] = deque(maxlen=window)

    def add(self, verdict: str) -> bool:
        if verdict in ("yes", "no"):
            self._window.append(verdict)
        return self.confirmed

    @property
    def confirmed(self) -> bool:
        return self._window.count("yes") >= self.required

    def reset(self) -> None:
        self._window.clear()
