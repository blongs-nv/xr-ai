# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A step's words: what it says and what it asks its models, as a schema the step type defines.

Every step type takes its words as an instance of its own schema, which extends :class:`StepWords` (``does``,
``start_line``, ``done_line``) with what only that type needs (a watch step's ``question`` and ``criteria``). The
step never reads YAML: an app keeps its words wherever it likes, and :func:`load_words` reads them from a YAML file
into the schema. Every schema forbids keys it does not name, so a misspelled key fails when the file is loaded::

    heat_water = WatchStep("heat_water", load_words(HERE / "steps/heat_water/words.yaml", WatchWords), ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

W = TypeVar("W", bound=BaseModel)


class StepWords(BaseModel):
    """What every step says; a step type's words extend it."""

    model_config = ConfigDict(extra="forbid")
    does: str | None = None
    """What the person does in this step, for the plan the conversation reads."""
    start_line: str | None = None
    """Spoken as the step starts ({placeholders}: fields or properties of its inputs)."""
    done_line: str | None = None
    """Spoken when the step finishes ({placeholders}: its inputs, and what its type names from its result)."""


def load_words(path: Path | str, schema: type[W], *, key: str | None = None) -> W:
    """The words in YAML file ``path`` as an instance of ``schema``; ``key`` (dotted: ``bed_clear.step``) picks them
    out of a file that holds several. A key the schema does not name is an error, with the file's name in it."""
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for part in key.split(".") if key else ():
        if not isinstance(raw, dict) or part not in raw:
            raise KeyError(f"{path}: no {key!r}")
        raw = raw[part]
    try:
        return schema.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path}{f' ({key})' if key else ''}: {exc}") from None
