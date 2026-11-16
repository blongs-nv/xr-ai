# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watching the camera over time: the looks a procedure step is made of.

A camera step asks: is it so, and is it really? These are the parts, each usable without the ones above it (and
outside a procedure, though the procedures are their only user so far):

=========================  ===========================================================================================
Outcomes                   Done, Needs, Stuck, Missing, Verdict, Report, Runnable, Outcome: what a look ends in and
                           how it says what it sees (it never speaks)
Pacing                     Eyes (fresh, distinct frames, at most one look per interval), Watching (the policy: how many
                           yes in how many looks, when to give up), FrameSource
Confirmation               Confirmation: k of the last n usable verdicts said yes
Plugs                      read, describe (observers from a perception role), by_choices, all_of (judges), Criteria,
                           Tier (an observer to escalate to), Observer, Judge, text_of
Looks                      Verify (watch until confirmed; escalate; give up), read_check (a structured read and a
                           judge in code), VisionRetrieval (read, look up in documents, fill the rest), Retrieved,
                           Source
=========================  ===========================================================================================

A one-off look is just ``await perceive.ask(frame, question, response_format=Model)``; these are for watching over
time. They use only the SDK's public names (``Frame``, ``FrameStream``, ``Role``, ``StructuredOutputError``,
``clock``, ``trace``). A look that returns an outcome can be exercised on a video window by itself
(``xr_ai_next.devkit``).
"""

from __future__ import annotations

from .confirmation import Confirmation
from .eyes import Eyes, FrameSource, Watching
from .outcomes import Done, Missing, Needs, Outcome, Report, Runnable, Stuck, Verdict
from .plugs import Criteria, Judge, Observer, Tier, all_of, by_choices, describe, read, text_of
from .retrieval import Retrieved, Source, VisionRetrieval
from .verify import Verify, read_check

__all__ = [
    "Confirmation",
    "Criteria",
    "Done",
    "Eyes",
    "FrameSource",
    "Judge",
    "Missing",
    "Needs",
    "Observer",
    "Outcome",
    "Report",
    "Retrieved",
    "Runnable",
    "Source",
    "Stuck",
    "Tier",
    "Verdict",
    "Verify",
    "VisionRetrieval",
    "Watching",
    "all_of",
    "by_choices",
    "describe",
    "read",
    "read_check",
    "text_of",
]
