# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard operating procedures: steps as plain async functions, and a runner that owns a procedure's run.

======================  =================================================================
Checks                  Check (what a visual check verifies: question, criteria, and
                        optionally the facts it needs and escalation guidance), Facts,
                        judge, Criteria, Judgment (positive / negative / unknown, from an
                        observation and a decision)
Steps                   verify (the fast loop: judge frames until N of the last M
                        confirm; reads the step's StepContext, never asks), timer,
                        identify (read into a schema, then derive a result, optionally
                        from retrieved text); each reports its state through ``on_state``
Escalation              gather (a side task beside a step: while the check's reports
                        show it cannot decide, a reasoning model reads them and asks
                        the person for what is missing, adding answers to the step's
                        StepContext, or tells them what to do, e.g. show the kettle),
                        escalation_for (one such judgement)
Runs                    SopRunner (a component: one run per participant, controls as a
                        router destination, a summary for the conversation), Run (what
                        the procedure function receives: ``await run.step(...,
                        alongside=[...])``, ``await run.ask(question)``)
======================  =================================================================

A procedure is ordinary Python::

    async def make_tea(run: Run) -> None:
        tea = await run.step("identify", lambda: identify(run.p.frames, ...), say="Show me the tea's label.")
        await run.step("heat_water", lambda: verify(run.p.frames, HOT, ..., on_state=run.report()), say="Heat it.")
        await run.step("steep", lambda: timer(180, on_state=run.report()), say="Steeping for three minutes.")

Every step function takes its inputs as arguments and knows nothing about the runner, the conversation or YAML, so
each can be used, tested and evaluated on its own (an evaluation calls :func:`judge` on recorded frames).
"""

from __future__ import annotations

from .escalate import Escalation, escalation_for, gather
from .identify import Identification, identify
from .judge import LABELS, Check, Criteria, Facts, Judgment, Label, StepContext, judge
from .runner import CONTROLS, MESSAGES, Run, SopRunner, StepRecord
from .timer import TimerState, timer
from .verify import Verification, verify

__all__ = [
    "CONTROLS",
    "Check",
    "Criteria",
    "Escalation",
    "Facts",
    "Identification",
    "Judgment",
    "LABELS",
    "Label",
    "MESSAGES",
    "Run",
    "SopRunner",
    "StepContext",
    "StepRecord",
    "TimerState",
    "Verification",
    "escalation_for",
    "gather",
    "identify",
    "judge",
    "timer",
    "verify",
]
