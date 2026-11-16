# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The interface between the runner and the existing harnesses.

Adapters live in ``evals/adapters`` and register under the ``xr_ai_eval.adapters`` entry-point group.
An adapter runs one existing harness and converts what it wrote into result records.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

from xr_ai_eval.profiles import Profile
from xr_ai_eval.schema import Record

ENTRY_POINT_GROUP = "xr_ai_eval.adapters"


@dataclass(frozen=True)
class AdapterContext:
    """What an adapter needs to run a harness and label its records."""

    repo_root: Path
    profile_name: str
    profile_path: Path | None
    profile: Profile | None
    commit: str
    run_id: str
    timestamp: str
    work_dir: Path


class Adapter(Protocol):
    """Runs one harness and returns schema records."""

    name: str

    def run(self, ctx: AdapterContext) -> list[Record]:
        """Run the harness and convert its output."""
        ...


def find_adapter(name: str) -> Adapter | None:
    """Return the registered adapter called ``name``, or ``None``."""
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        if entry.name == name:
            return entry.load()
    return None


def adapter_names() -> list[str]:
    """Names of all registered adapters."""
    return sorted(entry.name for entry in entry_points(group=ENTRY_POINT_GROUP))
