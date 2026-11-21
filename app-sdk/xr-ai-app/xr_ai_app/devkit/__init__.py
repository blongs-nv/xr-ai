# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local device: run any app without a headset or hub, the way a person (or a coding agent) would test it.

- :class:`LocalDevice`: start an app with an in-process device side; join, type turns, show images, play image
  folders and videos as timestamped frames; read every output with its time; read the trace.
- :class:`Scenario` and :func:`run_scenario`: a YAML file of timestamped user actions and checks, played and written
  up as ``report.json``, ``transcript.md`` and ``trace.jsonl``.
- :func:`serve`: the same device behind a local HTTP endpoint (``curl``-able).
- :func:`exercise`: run ONE piece of an app (any function, or a kit's piece: a router destination, a procedure step)
  on a window of a video, with context injected, through its real code; :class:`Exercise` says what it did. A
  package's target resolver finds the app's pieces with :func:`components`.
- :class:`VirtualTimeLoop` / :func:`run_virtual`: an event loop that skips idle waits (look intervals, timers, frame
  spacing) and still waits for real model calls, so a long window runs in moments.
- ``run_scenario(..., real_audio=True)``: fixed WAV/image input through real STT/TTS and a private hub,
  with an independent sender process and a performance section in the same report.
- CLI: ``python -m xr_ai_app.devkit run|serve|exercise ...`` (``--clock virtual`` for the skipping loop).

Roles are scripted (``FakeModels``) unless real models are asked for explicitly.
"""

from __future__ import annotations

from .device import LocalDevice, Output, fakes_from_file, image_frame, load_app
from .exercise import Exercise, Probe, Unit, components, exercise, resolve_target, target_kind
from .runner import Report, StepResult, run_scenario
from .scenario import Scenario, Step
from .server import serve
from .virtual_time import LatencyProfile, VirtualTimeLoop, run_virtual

__all__ = [
    "Exercise",
    "LocalDevice",
    "Output",
    "Probe",
    "Report",
    "Scenario",
    "Step",
    "StepResult",
    "LatencyProfile",
    "VirtualTimeLoop",
    "Unit",
    "components",
    "exercise",
    "fakes_from_file",
    "image_frame",
    "load_app",
    "resolve_target",
    "run_scenario",
    "run_virtual",
    "serve",
    "target_kind",
]
