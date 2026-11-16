# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501
"""The app core's lifecycle and profile contracts agree with their written form, and the package layering holds."""

from __future__ import annotations

import ast
import inspect
import re
import tomllib
from pathlib import Path

import pytest
from xr_ai_app import Component
from xr_ai_app._base import profiles, scope, topics
from xr_ai_interaction import UserQuery

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "app-sdk" / "xr-ai-app" / "docs"


def test_hooks_default_to_noops() -> None:
    for hook in ("on_start", "on_participant_enrolled", "on_participant_unenrolled", "on_stop"):
        assert inspect.iscoroutinefunction(getattr(Component, hook))


def test_standard_topics() -> None:
    names = {
        t.name
        for t in (
            topics.USER_QUERY,
            topics.PARTICIPANT_CONNECTED,
            topics.PARTICIPANT_DISCONNECTED,
            topics.INTERRUPTED,
            topics.OUTPUT,
        )
    }
    assert names == {
        "xr_ai_app.user_query",
        "xr_ai_app.participant_connected",
        "xr_ai_app.participant_disconnected",
        "xr_ai_app.interrupted",
        "xr_ai_app.output",
    }
    assert topics.USER_QUERY.message_type is UserQuery


def test_lifecycle_sequences_match_the_written_contract() -> None:
    text = (CONTRACT / "LIFECYCLE_CONTRACT.md").read_text(encoding="utf-8")
    for sequence in (scope.UNENROLL_SEQUENCE, scope.STOP_SEQUENCE):
        positions = [text.index(f"`{step}`") for step in sequence]
        assert positions == sorted(positions), sequence
    defaults = scope.ScopeTimeouts()
    for field, value in (("drain_s", 5.0), ("hook_s", 5.0), ("close_s", 10.0)):
        assert getattr(defaults, field) == value
        assert f"`{field}` = {value}" in text
    assert defaults.start_s is None and "`start_s` = `None`" in text
    from xr_ai_app._base.testing import LIFECYCLE_RULES

    assert len(LIFECYCLE_RULES) == 8
    for rule in LIFECYCLE_RULES:
        assert f"`{rule}`" in text, rule


def test_roles_match_the_profile_schema() -> None:
    text = (CONTRACT / "PROFILES_SCHEMA.md").read_text(encoding="utf-8")
    assert set(profiles.ROLE_CATEGORIES) == {
        "perceive",
        "decide",
        "extract",
        "answer",
        "ack",
        "embed",
        "speech_in",
        "speech_out",
    }
    for role in profiles.ROLE_CATEGORIES:
        assert f"`{role}`" in text
    assert not hasattr(profiles, "LEGACY_ROLE_ALIASES")


SDK_ROOT = "app-sdk"
"""The app SDK and its kits: built on the foundation packages (a layer above them)."""

SAMPLES_ROOT = "app-samples"
"""The app SDK's samples."""

CORE = "xr-ai-app"
"""The core package of the app SDK."""


PUBLIC_SDK_MODULES = {
    "",
    ".clock",
    ".trace",
    ".tools",
    ".llm",
    ".context",
    ".devkit",
    ".testing",
}
"""What a kit may import: the SDK's top level and its public modules (``xr_ai_app.output`` and the like are
internal files; their public names are re-exported at the top level)."""

KIT = "xr-ai-kit"
"""The kit: patterns built from the core's public names. It depends on the core alone."""

PUBLIC_KIT_MODULES = {"", ".conversation", ".routing", ".delegate"}
"""What another kit may import from ``xr_ai_kit`` (its top level and public modules), as it may from the SDK. The
kit itself is held to the SDK's public names like every kit, and imports no other kit."""


def test_kits_use_only_the_sdks_public_names() -> None:
    """A kit (a package under ``app-sdk/`` other than the SDK) builds on ``xr_ai_app`` and, unless it is
    the kit itself, ``xr_ai_kit``: it imports no other repository package, and from those two only their top level and
    public modules."""
    repository = re.compile(r"^\s*(?:from|import)\s+(xr_ai_\w+)((?:\.\w+)*)", re.M)
    packages = REPO / SDK_ROOT
    kits = sorted(d for d in packages.iterdir() if d.is_dir() and d.name != CORE)
    assert KIT in {k.name for k in kits}
    for kit in kits:
        allowed = {"xr_ai_app": PUBLIC_SDK_MODULES}
        if kit.name != KIT:
            allowed["xr_ai_kit"] = PUBLIC_KIT_MODULES
        for source in kit.rglob("*.py"):
            if ".venv" in source.parts:
                continue
            for package, path in repository.findall(source.read_text(encoding="utf-8")):
                if package == kit.name.replace("-", "_"):
                    continue
                assert package in allowed, f"{source}: imports {package}{path}"
                assert path in allowed[package], f"{source}: imports {package}{path}, not a public module"
        deps = tomllib.loads((kit / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
        repo_deps = [d for d in deps if re.match(r"xr-ai", d)]
        layer = r"xr-ai-app" if kit.name == KIT else r"xr-ai-(app|kit)"
        assert all(re.match(layer + r"(?![-\w])", d) for d in repo_deps), (kit.name, repo_deps)


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _defined_privates(tree: ast.AST) -> set[str]:
    """The ``_``-names a module defines: functions, classes, assigned names and assigned attributes."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return {n for n in names if _is_private(n)}


def _private_reaches(tree: ast.AST, own: set[str], sdk: set[str]) -> list[str]:
    """Where ``tree`` reaches a private (``_``, not dunder) name it may not: one the kit does not define (``own``),
    or one the SDK defines (``sdk``) on any receiver but ``self``/``cls``.

    A reach is an attribute (``app._components``) or a ``getattr``/``hasattr``/``setattr`` with a ``"_name"``
    literal. The kit's own objects may use the kit's own private names, so ``self._proc._dispatch`` passes; a name
    both define (``_say``, ``_frames``) passes only on ``self``/``cls``, where it is the kit's own.
    """

    def forbidden(name: str, receiver: ast.expr) -> bool:
        if not _is_private(name):
            return False
        on_self = isinstance(receiver, ast.Name) and receiver.id in ("self", "cls")
        return name not in own or (name in sdk and not on_self)

    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            hit = forbidden(node.attr, node.value)
        else:
            hit = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr", "setattr")
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and forbidden(node.args[1].value, node.args[0])
            )
        if hit:
            found.append(f"line {node.lineno}: {ast.unparse(node)}")
    return found


def _defined_in(root: Path) -> set[str]:
    return set().union(*(_defined_privates(ast.parse(s.read_text(encoding="utf-8"))) for s in _sources(root)))


def _sources(root: Path) -> list[Path]:
    return [s for s in sorted(root.rglob("*.py")) if ".venv" not in s.parts]


def test_the_private_reach_check_catches_attribute_and_getattr_forms() -> None:
    source = "class K:\n    def f(self):\n        self._mine = self._say = 1\n"
    source += "        return self._mine, getattr(self, '_mine'), self._say\n"
    source += "a = getattr(app, '_components', [])\nb = app._components\nc = k._mine\nd = p._say\ne = K.__name__\n"
    tree = ast.parse(source)
    found = _private_reaches(tree, _defined_privates(tree), {"_components", "_say"})
    assert found == ["line 5: getattr(app, '_components', [])", "line 6: app._components", "line 8: p._say"]


def test_kits_reach_no_private_names_of_sdk_objects() -> None:
    """A kit uses the SDK's public names only: no ``obj._x`` and no ``getattr(obj, "_x")`` for a private name the
    kit does not define itself, or that the SDK defines, except on ``self``/``cls`` (its own classes' names). For a
    kit built on ``xr_ai_kit``, the kit's private names count as the SDK's."""
    packages = REPO / SDK_ROOT
    sdk = _defined_in(packages / CORE / "xr_ai_app")
    kit_privates = _defined_in(packages / KIT / "xr_ai_kit")
    for kit in sorted(d for d in packages.iterdir() if d.is_dir() and d.name != CORE):
        own = _defined_in(kit)
        below = sdk if kit.name == KIT else sdk | kit_privates
        for source in _sources(kit):
            assert _private_reaches(ast.parse(source.read_text(encoding="utf-8")), own, below) == [], source


@pytest.mark.parametrize("package", ["xr_ai_app", "xr_ai_kit", "xr_ai_procedures", "xr_ai_kit.langchain"])
def test_every_module_of_the_sdk_and_kits_imports(package: str) -> None:
    """Every module imports: a name moved or renamed elsewhere fails here, not first in an app."""
    import importlib  # noqa: PLC0415
    import pkgutil  # noqa: PLC0415

    if package == "xr_ai_kit.langchain":
        pytest.importorskip("langchain_core")
    root = importlib.import_module(package)
    for info in pkgutil.walk_packages(root.__path__, package + "."):
        importlib.import_module(info.name)


def test_nothing_outside_the_app_sdk_depends_on_it() -> None:
    """The app SDK sits on top: no package outside ``app-sdk/`` and ``app-samples/`` (tests aside) depends on or
    imports it."""
    for pyproject in REPO.glob("**/pyproject.toml"):
        rel = pyproject.relative_to(REPO).parts
        if (
            rel[:1] in ((SDK_ROOT,), (SAMPLES_ROOT,), ("tests",))
            or {".venv", "node_modules"} & set(rel)
            or rel[:1] == (".claude",)
        ):
            continue
        deps = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {}).get("dependencies", [])
        assert not any(re.match(r"xr-ai-(app|kit|procedures)(?![-\w])", d) for d in deps), pyproject
    for source in (REPO / "agent-sdk").rglob("*.py"):
        if ".venv" in source.parts:
            continue
        text = source.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import) xr_ai_(app|kit|procedures)\b", text, re.M), source


def test_the_core_imports_no_kit() -> None:
    """The core sits under every kit: no module of ``xr_ai_app`` imports ``xr_ai_kit`` or another kit (the devkit
    finds a kit's exercise targets through the ``xr_ai_app.exercise_targets`` entry points)."""
    kits = re.compile(r"^\s*(?:from|import)\s+(xr_ai_(?:kit|procedures)\b)", re.M)
    for source in _sources(REPO / SDK_ROOT / CORE / "xr_ai_app"):
        assert kits.findall(source.read_text(encoding="utf-8")) == [], source
