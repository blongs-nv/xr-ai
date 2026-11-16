# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a long-running look ends in, and how it says what it sees.

A :class:`Runnable` is anything with ``async run(report) -> Done | Needs | Stuck``:

==================  ==========================================================================================
``Done(value)``     finished; ``value`` is its result
``Needs(fact)``     cannot go on without a fact only the user can give
``Stuck(reason)``   could not observe (``unseen``, ``unreadable``, ``unclear``), even after any escalation
==================  ==========================================================================================

What to do about ``Needs`` and ``Stuck`` is the caller's policy, never the runnable's. A judge answers
:class:`Missing` when it cannot judge without a fact. ``report(text)`` is how a running look says what it sees; it
never speaks. Returning an outcome (rather than only speaking) is what makes a piece testable on its own
(``xr_ai_next.devkit.exercise``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

__all__ = ["Done", "Missing", "Needs", "Outcome", "Report", "Runnable", "Stuck", "Verdict"]

Report = Callable[[str], Awaitable[None]]
"""What a running look calls with one line about what it sees or does."""

Verdict = Literal["yes", "no", "unknown"]
"""A judge's answer about one observation."""


@dataclass(frozen=True, slots=True)
class Done:
    """Finished; ``value`` is the result."""

    value: Any = None


@dataclass(frozen=True, slots=True)
class Needs:
    """Cannot go on without ``fact``, which only the user can give."""

    fact: str


@dataclass(frozen=True, slots=True)
class Stuck:
    """Could not observe, whatever it tried."""

    reason: Literal["unseen", "unreadable", "unclear"]


@dataclass(frozen=True, slots=True)
class Missing:
    """A judge's answer when it cannot judge without ``fact``."""

    fact: str


Outcome = Done | Needs | Stuck


class Runnable(Protocol):
    """One run of a watch, check or wait, from frozen parameters to an outcome (``Verify``, ``VisionRetrieval``, a
    procedure step, anything of your own)."""

    async def run(self, report: Report) -> Outcome: ...
