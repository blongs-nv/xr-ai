# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The words and the order of a procedure (an SOP), loaded from YAML.

What each step *does* is code (a :class:`~.step.StepDef` per step id); the SOP says what the steps are called, in
what order, and what the procedure says. Only ``steps`` is required::

    name: printer pre-flight
    steps:
      - id: bed_clear
        instruction: clear everything off the print bed     # what the user should be doing (the summary says it)
        enter: First, make sure the print bed is clear.     # said when the step starts
        done: The bed is clear.                             # said, with the next step's `enter`, when it is done
        skipped: ""                                          # the same, when it was skipped (optional)
        unable: confirm                                      # show | confirm | skip: when it cannot observe
        confirm: Is the print bed clear?                     # the yes/no question for `unable: confirm`
        asks: {view: ...}                                    # this step's own questions (override `asks`)
    messages: {...}   # the procedure's own words; built-in defaults for any left out (DEFAULT_MESSAGES)
    asks: {...}       # the questions it asks by what is missing (DEFAULT_ASKS)

Lines may hold ``{name}`` placeholders, filled from the words the steps provide (``StepDef.words``), plus
``{procedure}`` (the SOP's name) and, where they apply, ``{told}``, ``{instruction}`` and ``{question}``. A
placeholder nobody provides is left as written. Unknown keys in a step are rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["DEFAULT_ASKS", "DEFAULT_MESSAGES", "Sop", "StepSpec"]

DEFAULT_MESSAGES: dict[str, str] = {
    "start": "Okay, starting {procedure}.",
    "stopped": "Okay, I've stopped.",
    "next": "Okay, moving on.",
    "back": "Okay, going back.",
    "rejected": "Nothing is running yet. Say start to begin {procedure}.",
    "unsure": "I'm not sure what you want me to do. Could you say it again?",
    "noted": "Got it: {told}.",
    "noted_late": "Got it: {told}. That step is already done; say go back if you want to do it again with that.",
    "gave_up": "I couldn't verify that step, so I'm moving on.",
    "complete": "That was the last step.",
    "failed": "Something went wrong, so I've stopped {procedure}. Say start to try again.",
}
"""What the procedure says for its own events; ``Sop.messages`` overrides any of them."""

DEFAULT_ASKS: dict[str, str] = {
    "view": (
        "I can't tell what the camera shows. Please {instruction}, with it in view of the camera, "
        "or say next to skip this step."
    ),
    "confirm": "I can't verify this with the camera. {question} Say yes or no, or say next to skip this step.",
    "retry": "I had trouble with this step. Say next to skip it, or tell me more.",
}
"""The questions the procedure asks by what is missing; ``Sop.asks`` and a step's ``asks`` override them."""


class StepSpec(BaseModel):
    """One step's words (see the module docstring)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    enter: str
    done: str
    skipped: str = ""
    asks: dict[str, str] = Field(default_factory=dict)
    unable: Literal["show", "confirm", "skip"] = "show"
    confirm: str = ""

    @model_validator(mode="after")
    def _confirm_needs_a_question(self) -> StepSpec:
        if self.unable == "confirm" and not self.confirm:
            raise ValueError(f"step {self.id}: unable=confirm needs a confirm question")
        return self


class Sop(BaseModel):
    """A whole SOP. ``messages`` and ``asks`` are merged over the built-in defaults; other keys are the
    procedure's own words."""

    model_config = ConfigDict(extra="allow")

    name: str = "the procedure"
    steps: list[StepSpec]
    messages: dict[str, str] = Field(default_factory=dict)
    asks: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _complete(self) -> Sop:
        if not self.steps:
            raise ValueError("an SOP needs at least one step")
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate step ids: {ids}")
        self.messages = {**DEFAULT_MESSAGES, **self.messages}
        self.asks = {**DEFAULT_ASKS, **self.asks}
        return self

    @classmethod
    def load(cls, path: str | Path) -> Sop:
        """Read and validate an SOP file."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def step(self, step_id: str) -> StepSpec:
        """The step with id ``step_id``."""
        for spec in self.steps:
            if spec.id == step_id:
                return spec
        raise KeyError(step_id)
