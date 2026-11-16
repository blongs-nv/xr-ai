# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a running procedure tells the rest of the app: one :class:`ProcedureEvent` per transition.

Subscribe with ``procedure.on(kind, handler)`` (``"*"`` for all); a handler is ``handler(event, p)``, sync or async,
and is awaited in order, so keep it short and ``p.spawn`` anything long. A handler that raises is traced and never
stops the run.

==================  ==========================================================================================
``run_start``       a run began (``data["steps"]``: the step ids in order)
``step_start``      a step began (``step``; ``data["index"]``)
``step_done``       a step ended (``data["how"]``: ``done``, ``confirmed`` or ``skipped``; ``data["value"]``)
``step_needs``      a step needs a fact only the user can give (``data["fact"]``)
``step_stuck``      a step could not observe (``data["reason"]``, ``data["policy"]``)
``step_failed``     a step raised (``data["error"]``, ``data["policy"]``)
``ask``             the procedure asked the user and waits (``data["about"]``, ``data["text"]``)
``back``            it went back to ``step``
``run_end``         the run ended (``data["status"]``: ``completed``, ``stopped`` or ``failed``; ``data["outcomes"]``)
==================  ==========================================================================================
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from xr_ai_app import Participant

__all__ = ["EventKind", "Listener", "ProcedureEvent"]

EventKind = Literal[
    "run_start", "step_start", "step_done", "step_needs", "step_stuck", "step_failed", "ask", "back", "run_end"
]


@dataclass(frozen=True)
class ProcedureEvent:
    """One transition of a run (see the module docstring)."""

    kind: EventKind
    procedure: str
    participant: str
    step: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


Listener = Callable[[ProcedureEvent, "Participant"], Any]
"""``(event, p) -> None``, sync or async."""
