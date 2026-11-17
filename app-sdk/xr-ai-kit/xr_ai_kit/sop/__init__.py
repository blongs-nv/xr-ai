# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard operating procedures: steps, a procedure that runs them in order, and a runner that runs it with a person.

Three layers, each with one job:

======================  =================================================================
Steps                   WatchStep (the camera confirms something; the person helps when
(:mod:`.steps`)         it cannot), ReadStep (read a label, derive settings), WaitStep (a
                        countdown). Each is ``run(io)`` over a narrow StepIO (frames, say,
                        ask, report, situation), plus how an evaluation case of it is
                        scored (``Expect``, ``score``)
The procedure           the app's ``async def procedure(run)``: which steps, in what
                        order, what passes between them (``await run.step(heat)``);
                        Wording loads its words from YAML and builds the steps
The runner              SopRunner (a component: one run per participant, controls as a
                        router destination, the run's summary for the conversation); Run
                        (what the procedure receives, and what a step touches)
======================  =================================================================

Under the steps: :func:`judge` (one frame against a :class:`Check`), :func:`verify` (the fast check), :func:`gather`
(the side escalation), :func:`identify`, :func:`timer`. :mod:`.eval` runs one step on recorded frames and a scripted
person, on the devkit's virtual clock, and scores it with the step's own ``score``; it is imported only to evaluate.

A procedure::

    WORDS = Wording.load("steps.yaml")
    fill = WORDS.watch("fill_water", roles)
    steep = WORDS.wait("steep")

    async def make_tea(run: Run) -> None:
        if await run.step(fill):
            await run.say(WORDS.line("water_seen"))
        await run.step(steep.lasting(120))
"""

from __future__ import annotations

from .escalate import Escalation, escalation_for, gather
from .identify import Identification, identify
from .judge import LABELS, Check, Criteria, Facts, Judgment, Label, StepContext, judge
from .runner import CONTROLS, MESSAGES, Run, SopRunner, StepRecord
from .step import Frames, Line, Roles, Step, StepIO, StepTrace
from .steps import ReadStep, WaitStep, WatchSettings, WatchStep
from .timer import TimerState, timer
from .verify import Verification, verify
from .wording import StepWords, Wording

__all__ = [
    "CONTROLS",
    "Check",
    "Criteria",
    "Escalation",
    "Facts",
    "Frames",
    "Identification",
    "Judgment",
    "LABELS",
    "Label",
    "Line",
    "MESSAGES",
    "ReadStep",
    "Roles",
    "Run",
    "SopRunner",
    "Step",
    "StepContext",
    "StepIO",
    "StepRecord",
    "StepTrace",
    "StepWords",
    "TimerState",
    "Verification",
    "WaitStep",
    "WatchSettings",
    "WatchStep",
    "Wording",
    "escalation_for",
    "gather",
    "identify",
    "judge",
    "timer",
    "verify",
]
