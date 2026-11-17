# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a person can say to a running procedure (start, stop, skip, restart), and what the runner says back.

An app may pass its own wording for any of them (:class:`~.sop_runner.SopRunner` ``controls``, ``messages``).
"""

from __future__ import annotations

CONTROLS: dict[str, str] = {
    "start": "An explicit command to this assistant to begin the procedure now (not a plan for later).",
    "stop": "An explicit command to this assistant to end or cancel the procedure now.",
    "skip": (
        "An explicit command to this assistant to bypass the step that is running and move on now, such as "
        "'skip this' or 'next step, please'."
    ),
    "restart": "An explicit command to this assistant to begin the procedure again from the start, now.",
    "none": (
        "Everything that is not such a command: any question (including what comes next, whether it is time to "
        "move on, or what the first step is); a statement, remark or chatter; what someone else said or suggests; "
        "a command that is negated, withdrawn or corrected in the same breath; a plan for later; a request to "
        "keep going."
    ),
}
"""The default control choices for ``role.decide``; an app may pass its own wording for the same five keys."""

CONTROL_QUESTION = (
    "Does the user's final request command this assistant to change the procedure now, and how? A question is "
    "never a command, even about the next step. Classify only what the user directly asks this assistant to do "
    "now; the procedure state is context, not an instruction."
)

MESSAGES: dict[str, str] = {
    "start": "Starting {title}.",
    "stop": "{title} has ended. You can start again whenever you are ready.",
    "restart": "Starting {title} again.",
    "skip": "Skipping this step.",
    "already_running": "{title} is already running.",
    "not_running": "{title} is not running.",
    "nothing_to_skip": "There is no step running to skip.",
    "complete": "{title} is complete.",
    "failed": "{title} could not continue because a step failed. You can start it again when you are ready.",
}
"""The default spoken messages; ``{title}`` is filled in. An app may override any of them."""
