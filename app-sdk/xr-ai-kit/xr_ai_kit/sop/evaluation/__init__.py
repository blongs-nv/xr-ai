# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one step at a time: the real step, on a recorded scene and a scripted person.

================  ==========================================================================================
``cases``         case files, :func:`run_case`, the command line (:func:`main`); finds each step type's scoring
``scene``         a recording, and a scene from it played as the step's camera frames
``scripted_io``   the scripted person (:class:`ScriptedIO`: a step's and an escalator's io)
``step_trace``    what a case records of the step (:class:`StepTrace`), which the scoring reads
================  ==========================================================================================

What every step shares lives here; how a case is scored lives with what is scored, in its own scoring module
(``step_types/watch/watch_scoring.py``, ..., and ``escalation/escalation_scoring.py`` for an escalator beside a
step). Nothing here looks inside a step. Imported only to evaluate;
an app running a procedure never imports it.

An app's evaluation script::

    from xr_ai_kit.sop import evaluation
    raise SystemExit(evaluation.main(None, build=build_steps, cases=HERE / "steps", fakes_from=fakes_from_file))
"""

from __future__ import annotations

from .cases import Build, Case, CaseFile, CaseResult, Scoring, Scorings, main, report, run_case, run_files, scoring_for
from .scene import Recording, SceneFrames, Shot, segments
from .scripted_io import Person, ScriptedIO
from .step_trace import Line, StepTrace, mentions

__all__ = [
    "Build",
    "Case",
    "CaseFile",
    "CaseResult",
    "Line",
    "Person",
    "Recording",
    "SceneFrames",
    "Scoring",
    "Scorings",
    "ScriptedIO",
    "Shot",
    "StepTrace",
    "main",
    "mentions",
    "report",
    "run_case",
    "run_files",
    "scoring_for",
    "segments",
]
