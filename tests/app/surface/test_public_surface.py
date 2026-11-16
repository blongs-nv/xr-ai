# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDK's public surface is exactly what it documents.

Each core class exposes the public names listed here and in the package README's "Classes" table, no more and no
fewer; the top level (of the core, and of the kit ``xr_ai_kit``) exports the names of the module docstring's table.
A name that is not documented (for example a method inherited from an earlier layer) fails these tests; adding one
means documenting it in both places.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

import pytest
import xr_ai_app
import xr_ai_kit

_README = Path(xr_ai_app.__file__).resolve().parents[1] / "README.md"

SURFACE: dict[str, set[str]] = {
    "App": {
        "add",
        "from_config",
        "on_start",
        "on_turn",
        "participants",
        "role",
        "run",
        "settings",
        "start",
        "state",
        "stop",
        "test_session",
        "trace",
    },
    "Component": {"app", "on_participant_enrolled", "on_participant_unenrolled", "on_start", "on_stop", "spawn"},
    "Participant": {
        "artifacts",
        "cancel",
        "connected",
        "frames",
        "id",
        "name",
        "reply",
        "say",
        "session",
        "spawn",
        "state",
        "withdraw",
    },
    "FrameStream": {"current", "every"},
    "Session": {"id", "state"},
    "Turn": {"id", "replace", "state", "text", "timestamp_us"},
}


def _public(cls: type) -> set[str]:
    return {n for n in dir(cls) if not n.startswith("_")}


@pytest.mark.parametrize("name", sorted(SURFACE))
def test_each_core_class_exposes_exactly_its_documented_names(name: str) -> None:
    cls = getattr(xr_ai_app, name)
    assert _public(cls) == SURFACE[name]


def test_the_readme_documents_the_same_class_surface() -> None:
    rows = re.findall(r"^\| `(\w+)` \| (.+) \|$", _README.read_text(encoding="utf-8"), flags=re.M)
    documented = {cls: set(re.findall(r"`(?:\w+\.)?(\w+)`", names)) for cls, names in rows if cls in SURFACE}
    assert documented == SURFACE


@pytest.mark.parametrize("package", [xr_ai_app, xr_ai_kit], ids=lambda m: m.__name__)
def test_the_top_level_exports_exactly_the_docstring_table(package: ModuleType) -> None:
    lines = (package.__doc__ or "").splitlines()
    rules = [i for i, line in enumerate(lines) if line.startswith("=" * 22)]
    assert len(rules) == 2, "the module docstring has one table"
    right = " ".join(line[24:] for line in lines[rules[0] + 1 : rules[1]])  # names column (the first is 22 wide)
    names = re.findall(r"\w+", re.sub(r"\([^()]*\)", "", right))  # asides in parentheses are not exports
    assert len(names) == len(set(names))
    assert set(package.__all__) == set(names)
