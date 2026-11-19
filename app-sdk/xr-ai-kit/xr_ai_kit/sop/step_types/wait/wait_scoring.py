# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How an evaluation case of a wait step is scored: what a case may expect (:class:`WaitExpect`) and :func:`score`.

Evaluation only: the step itself (:mod:`.wait_step`) never imports this.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ...evaluation.step_trace import StepTrace


class WaitExpect(BaseModel):
    """What an evaluation case may expect of a wait step."""

    model_config = ConfigDict(extra="forbid")
    completes: bool | None = None
    """Whether the countdown finished within the scene."""


def score(trace: StepTrace, expect: WaitExpect) -> list[str]:
    """What is wrong with ``trace`` against ``expect`` (empty: the case passed)."""
    if expect.completes is not None and trace.completed != expect.completes:
        return ["completed" if trace.completed else "did not complete"]
    return []


Expect = WaitExpect  # the name evaluation looks up
