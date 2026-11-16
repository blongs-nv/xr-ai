# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the build-your-app and run-demo skills point at exists (files and docs pages)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SKILLS = ("build-your-app", "run-demo")
TOP_LEVEL = {p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")}
TICKS = re.compile(r"`([^`\n]+)`")
LINKS = re.compile(r"\]\(([^)\s]+)\)")


def text(skill: str) -> str:
    return (ROOT / "skills" / skill / "SKILL.md").read_text()


def repo_paths(source: str) -> list[str]:
    found = []
    for target in TICKS.findall(source):
        if any(c in target for c in " <>*") or ":" in target:
            continue
        if target.split("/")[0] in TOP_LEVEL and "/" in target:
            found.append(target.rstrip("/"))
    return found


@pytest.mark.parametrize("skill", SKILLS)
def test_referenced_repository_files_exist(skill: str) -> None:
    for target in repo_paths(text(skill)):
        assert (ROOT / target).exists(), f"{skill}: {target} does not exist"


@pytest.mark.parametrize("skill", SKILLS)
def test_relative_links_resolve(skill: str) -> None:
    for target in LINKS.findall(text(skill)):
        if "://" not in target and not target.startswith("#"):
            assert (ROOT / "skills" / skill / target.split("#")[0]).resolve().exists(), target
