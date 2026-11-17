# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How an evaluation case of a watch step is scored: what a case may expect (:class:`WatchExpect`) and :func:`score`.

Evaluation only: the step itself (:mod:`.watch_step`) never imports this.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ...evaluation.step_trace import StepTrace, mentions


class Window(BaseModel):
    """A time window on the evaluation's scene clock."""

    model_config = ConfigDict(extra="forbid")
    after: float | None = None
    before: float | None = None


class WatchExpect(BaseModel):
    """What an evaluation case may expect of a watch step (each key optional)."""

    model_config = ConfigDict(extra="forbid")
    completes: bool | Window | None = None
    """``true``/``false``, or completes within a window of scene time."""
    asks: list[str] | None = None
    """It asks the person something that mentions one of these words (``[]``: any question)."""
    says: list[str] | None = None
    """It tells the person something (not a question) that mentions one of these words (``[]``: anything)."""
    quiet: bool = False
    """It neither asks nor tells the person anything."""


def score(trace: StepTrace, expect: WatchExpect) -> list[str]:
    """What is wrong with ``trace`` against ``expect`` (empty: the case passed)."""
    problems: list[str] = []
    done, at = trace.completed, trace.completed_at_s
    if isinstance(expect.completes, bool) and done != expect.completes:
        problems.append(f"completed at {at:.0f} s" if done and at is not None else "did not complete")
    elif isinstance(expect.completes, Window):
        w = expect.completes
        if not done or at is None:
            problems.append("did not complete")
        elif (w.after is not None and at < w.after) or (w.before is not None and at > w.before):
            problems.append(f"completed at {at:.0f} s, expected within [{w.after}, {w.before}]")
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


Expect = WatchExpect
"""What a case of this step type may expect (the name evaluation looks up)."""
