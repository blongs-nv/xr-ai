# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Python code blocks of the new docs pages are real.

Every ``python`` block in the pages below must parse. A block preceded by an HTML comment
``<!-- docs-snippet: run group=G cwd=yaml|repo [sample=<path>] -->`` is executed: the blocks of one group on a page
are joined and executed as one module (``__name__`` is not ``__main__``, so ``app.run()`` guards stay off). With
``cwd=yaml`` a copy of ``fixtures/profile_app/`` (an app configuration with a ``local`` profile) is the working
directory, so ``App.from_config("app.yaml")`` loads that configuration. When the module defines ``app`` it must
pass the lifecycle contract suite, and when it defines ``async def check(app)`` that function runs on the test kit
with fakes. ``sample=<path>`` (relative to ``agent-samples/``) requires the block to equal that sample file apart
from its header and ``from_config`` path.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from xr_ai_next._base.scope import ScopeTimeouts
from xr_ai_next._base.testing import run_lifecycle_contract

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "source"
PAGES = (
    "guides/choosing-models.md",
    "guides/profiles-and-deployment.md",
)
APP_YAML = Path(__file__).resolve().parent / "fixtures" / "profile_app"
FENCE = re.compile(r"(?:<!-- docs-snippet: (?P<tag>[^>]*?) -->\n)?```python\n(?P<code>.*?)```", re.S)


def blocks(page: str) -> list[tuple[dict[str, str], str]]:
    found = []
    for m in FENCE.finditer((DOCS / page).read_text()):
        tag = dict(t.split("=", 1) if "=" in t else (t, "") for t in (m["tag"] or "").split())
        found.append((tag, m["code"]))
    return found


def existing_pages() -> list[str]:
    return [p for p in PAGES if (DOCS / p).exists()]


def test_the_new_pages_exist_and_have_runnable_code() -> None:
    assert existing_pages() == list(PAGES)
    assert sum("run" in tag for p in PAGES for tag, _ in blocks(p)) >= 1


@pytest.mark.parametrize("page", existing_pages())
def test_every_python_block_parses(page: str) -> None:
    for tag, code in blocks(page):
        ast.parse(code, filename=f"{page}")
        assert "run" in tag or "sample" not in tag


def normalized_sample(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    body = [n for n in body if not (isinstance(n, ast.ImportFrom) and n.module == "pathlib")]
    text = "\n".join(ast.unparse(n) for n in body)
    text = re.sub(r"App\.from_config\(.*?, enrollment=", "App.from_config('app.yaml', enrollment=", text, flags=re.M)
    return [line.strip() for line in text.splitlines() if line.strip()]


@pytest.mark.parametrize("page", existing_pages())
def test_blocks_tagged_as_a_sample_equal_the_sample_file(page: str) -> None:
    for tag, code in blocks(page):
        if "sample" in tag:
            doc = ast.unparse(ast.parse(code))
            want = "\n".join(normalized_sample(ROOT / "agent-samples" / tag["sample"]))
            assert [ln.strip() for ln in doc.splitlines() if ln.strip()] == want.splitlines()


def groups(page: str) -> dict[str, tuple[str, str]]:
    joined: dict[str, tuple[str, list[str]]] = {}
    for tag, code in blocks(page):
        if "run" in tag:
            cwd, parts = joined.setdefault(tag["group"], (tag.get("cwd", "yaml"), []))
            assert cwd == tag.get("cwd", "yaml")
            parts.append(code)
    return {g: (cwd, "\n".join(parts)) for g, (cwd, parts) in joined.items()}


CASES = [(p, g) for p in existing_pages() for g in groups(p)]


@pytest.mark.parametrize(("page", "group"), CASES)
async def test_runnable_groups_execute(page: str, group: str, tmp_path: Path, monkeypatch) -> None:
    cwd, source = groups(page)[group]
    where = ROOT if cwd == "repo" else tmp_path
    if cwd == "yaml":
        shutil.copytree(APP_YAML, tmp_path / "yaml")
        where = tmp_path / "yaml"
    monkeypatch.chdir(where)

    def run_source() -> dict:
        scope = {"__name__": f"docs_snippet_{group}"}
        exec(compile(source, f"{page}:{group}", "exec"), scope)  # noqa: S102 - trusted repository docs
        return scope

    scope = run_source()
    app = scope.get("app")
    if app is None:
        return

    def fresh_app():
        built = run_source()["app"]
        built._config = replace(built._config, timeouts=ScopeTimeouts(drain_s=0.5, hook_s=0.5, close_s=0.5))
        return built

    await run_lifecycle_contract(fresh_app)
    if "check" in scope:
        await scope["check"](run_source()["app"])
    assert os.getcwd() == str(where)
