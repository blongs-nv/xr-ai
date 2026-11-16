# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``StepDef``: what one step declares to the procedure.

``build(p, run, spec) -> (step, key)`` is a pure function of the run: it returns a step (``Verify``, ``Timer``,
anything with ``async run(report)``) and a key, the values the step was built from. When facts change the supervisor
builds again; if the key differs, the running step is replaced.

A step owns its facts: which facts the user may tell it (a :class:`~.run.Facts` model), how each is said back, and
which only matter if the step runs again. The procedure reads the user's words once, against the facts of all
steps, and hands each fact to the step that owns it. ``words(run)`` gives the values the SOP's ``{name}``
placeholders are filled from (for example the temperature a step decided).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel
    from xr_ai_app import Participant

    from .perception import Runnable
    from .run import Run
    from .spec import StepSpec

__all__ = ["Builder", "StepDef"]

Builder = Callable[["Participant", "Run", "StepSpec"], "tuple[Runnable, Hashable]"]


@dataclass(frozen=True)
class StepDef:
    """One step: its id (as in the SOP), its builder, and the facts it owns.

    - ``facts``: what the user may tell this step (every field optional);
    - ``phrases``: how each fact is said back, ``{"display_unit": "the display shows {}"}``;
    - ``reruns``: facts that only take effect if this step runs again (told after it is done, the procedure says
      to go back);
    - ``needs``: facts this step may report as ``Needs``; each needs a question (the step's ``asks`` in the SOP, or
      the global ones), checked when the procedure is built;
    - ``words``: ``(run) -> {placeholder: value}`` for the SOP's lines.
    """

    id: str
    build: Builder
    facts: type[BaseModel] | None = None
    phrases: Mapping[str, str] = field(default_factory=dict)
    reruns: frozenset[str] = frozenset()
    needs: frozenset[str] = frozenset()
    words: Callable[[Run], Mapping[str, Any]] | None = None
