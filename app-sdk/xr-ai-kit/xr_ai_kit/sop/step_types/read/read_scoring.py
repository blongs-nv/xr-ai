# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How an evaluation case of a read step is scored: what a case may expect (:class:`ReadExpect`) and :func:`score`.

Evaluation only: the step itself (:mod:`.read_step`) never imports this.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ...evaluation.step_trace import StepTrace, mentions


class ReadExpect(BaseModel):
    """What an evaluation case may expect of a read step (each key optional)."""

    model_config = ConfigDict(extra="forbid")
    completes: bool | None = None
    reads: dict[str, list[str]] | None = None
    """Per field of what was read: the value mentions one of these words (``[]``: the field is not empty)."""
    derives: dict[str, list[str]] | None = None
    """The same, per field of the derived result."""


def score(trace: StepTrace, expect: ReadExpect) -> list[str]:
    """What is wrong with ``trace`` against ``expect`` (empty: the case passed)."""
    problems: list[str] = []
    if expect.completes is not None and trace.completed != expect.completes:
        problems.append("completed" if trace.completed else "did not complete")
    result = trace.result
    for name, wanted, model in (
        ("read", expect.reads, getattr(result, "visual", None)),
        ("derived", expect.derives, getattr(result, "result", None)),
    ):
        for key, words in (wanted or {}).items():
            value = getattr(model, key, None)
            if value is None or not mentions(str(value), words):
                problems.append(f"{name} {key}={value!r}, expected one of {words or 'any value'}")
    return problems


Expect = ReadExpect
"""What a case of this step type may expect (the name evaluation looks up)."""
