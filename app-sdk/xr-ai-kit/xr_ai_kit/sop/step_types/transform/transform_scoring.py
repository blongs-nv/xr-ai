# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How an evaluation case of a transform step is scored: what a case may expect (:class:`TransformExpect`) and
:func:`score`.

Evaluation only: the step itself (:mod:`.transform_step`) never imports this.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ...evaluation.step_trace import StepTrace, mentions


class TransformExpect(BaseModel):
    """What an evaluation case may expect of a transform step (each key optional)."""

    model_config = ConfigDict(extra="forbid")
    completes: bool | None = None
    output: dict[str, list[str]] | None = None
    """Per field of the output: the value mentions one of these words (``[]``: the field is not empty)."""
    asks: list[str] | None = None
    """It asks the person something that mentions one of these words (``[]``: any question)."""
    quiet: bool = False
    """It asks the person nothing."""


def score(trace: StepTrace, expect: TransformExpect) -> list[str]:
    """What is wrong with ``trace`` against ``expect`` (empty: the case passed)."""
    problems: list[str] = []
    if expect.completes is not None and trace.completed != expect.completes:
        problems.append("completed" if trace.completed else "did not complete")
    for key, words in (expect.output or {}).items():
        value = getattr(trace.result, key, None)
        if value is None or not mentions(str(value), words):
            problems.append(f"output {key}={value!r}, expected one of {words or 'any value'}")
    asked = trace.asked(by="step")
    if expect.asks is not None and not any(mentions(line.text, expect.asks) for line in asked):
        got = "; ".join(repr(line.text) for line in asked) or "nothing"
        problems.append(f"asked {got}, expected a question about {expect.asks or 'anything'}")
    if expect.quiet and asked:
        problems.append("asked " + "; ".join(repr(line.text) for line in asked) + ", expected nothing")
    return problems


Expect = TransformExpect
"""What a case of this step type may expect (the name evaluation looks up)."""
