# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The new docs and skills agree with the public surface (no deprecated names, migration links)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "source"
NEW_PAGES = (
    "guides/choosing-models.md",
    "guides/profiles-and-deployment.md",
)
SKILLS = ("build-your-app", "run-demo")
OLD_PACKAGE_PAGES = sorted(p.relative_to(DOCS).as_posix() for p in (DOCS / "reference").glob("agent-sdk-*.md"))
DEPRECATED_SPELLINGS = (
    re.compile(r"\bAgent\(\s*tools="),
    re.compile(r"\.subscriber\(\)"),
    re.compile(r"\bLLMSpec\(\s*base_url="),
)


def sources() -> list[tuple[str, Path]]:
    found = [(page, DOCS / page) for page in NEW_PAGES]
    found += [(f"skills/{skill}", ROOT / "skills" / skill / "SKILL.md") for skill in SKILLS]
    return found


@pytest.mark.parametrize(("name", "path"), sources())
def test_no_deprecated_spelling_is_taught(name: str, path: Path) -> None:
    body = path.read_text()
    for pattern in DEPRECATED_SPELLINGS:
        assert not pattern.search(body), f"{name}: {pattern.pattern}"


def test_old_package_pages_link_the_migration_entry() -> None:
    assert OLD_PACKAGE_PAGES
    for page in [*OLD_PACKAGE_PAGES, "components/agent-sdk.md"]:
        assert re.search(r"\{doc\}`(?:[^`]*<)?(?:/reference/)?migrations>?`", (DOCS / page).read_text()), page


def test_migration_entry_exists_and_names_the_deprecations() -> None:
    body = (DOCS / "reference" / "migrations.md").read_text()
    assert "Internal and duplicate names deprecated" in body
    for name in ("Agent(tools=[...])", "manager.subscriber()"):
        assert name in body
