# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How an evaluation case of a wait step is scored: what a case may expect (:class:`WaitExpect`) and :func:`score`.

Evaluation only: the step itself (:mod:`.wait_step`) never imports this.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..evaluation.step_trace import StepTrace


class WaitExpect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    completes: bool | None = None


def score(trace: StepTrace, expect: WaitExpect) -> list[str]:
    if expect.completes is not None and trace.completed != expect.completes:
        return ["completed" if trace.completed else "did not complete"]
    return []


Expect = WaitExpect
"""What a case of this step type may expect (the name evaluation looks up)."""
