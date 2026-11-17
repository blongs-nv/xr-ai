# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How an evaluation case scores an escalator: what a case may expect of it (:class:`EscalationExpect`) and
:func:`score`. A case of an escalated step has these keys beside its step's own (``completes`` for a watch step).

Evaluation only: the escalator never imports this.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..evaluation.step_trace import StepTrace, mentions


class EscalationExpect(BaseModel):
    """What an evaluation case may expect of the escalator (each key optional)."""

    model_config = ConfigDict(extra="forbid")
    asks: list[str] | None = None
    """It asks the person something that mentions one of these words (``[]``: any question)."""
    says: list[str] | None = None
    """It tells the person something (not a question) that mentions one of these words (``[]``: anything)."""
    quiet: bool = False
    """It neither asks nor tells the person anything."""


def score(trace: StepTrace, expect: EscalationExpect) -> list[str]:
    """What is wrong with ``trace`` against ``expect`` (empty: the case passed)."""
    problems: list[str] = []
    asked, said = trace.asked(), trace.said()
    if expect.asks is not None and not any(mentions(line.text, expect.asks) for line in asked):
        got = "; ".join(repr(line.text) for line in asked) or "nothing"
        problems.append(f"asked {got}, expected a question about {expect.asks or 'anything'}")
    if expect.says is not None and not any(mentions(line.text, expect.says) for line in said):
        got = "; ".join(repr(line.text) for line in said) or "nothing"
        problems.append(f"said {got}, expected an instruction about {expect.says or 'anything'}")
    if expect.quiet and (asked or said):
        problems.append("said " + "; ".join(repr(line.text) for line in asked + said) + ", expected nothing")
    return problems


Expect = EscalationExpect
"""What a case may expect of the escalator (the name evaluation looks up)."""
