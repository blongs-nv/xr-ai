# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one step at a time: the real step, on a recorded scene and a scripted person.

What every step shares lives here; how a case of a given step type is scored lives with the step (its ``Expect`` and
``score``, in :mod:`..steps`). Nothing here looks inside a step: it plays the scene (:mod:`.scene`), answers as the
person (:mod:`.io`), records what the step did (a :class:`~..step.StepTrace`) and hands it to the step's ``score``
(:mod:`.run`). Imported only to evaluate; an app running a procedure never imports it.

An app's evaluation script::

    from xr_ai_kit.sop import eval as step_eval
    raise SystemExit(step_eval.main(None, build=build_steps, cases=HERE / "steps", fakes_from=fakes_from_file))
"""

from __future__ import annotations

from .io import Person, ScriptedIO
from .run import Build, Case, CaseFile, CaseResult, main, report, run_case, run_files
from .scene import Recording, SceneFrames, Shot, segments

__all__ = [
    "Build",
    "Case",
    "CaseFile",
    "CaseResult",
    "Person",
    "Recording",
    "SceneFrames",
    "ScriptedIO",
    "Shot",
    "main",
    "report",
    "run_case",
    "run_files",
    "segments",
]
