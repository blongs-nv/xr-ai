# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Snapshot the public API (``__all__``) of every package in the repository.

``api-snapshot.json`` records, for each package ``__init__.py`` tracked by git
that defines ``__all__``, the sorted exported names. ``--check`` fails when the
recorded and actual surface differ in either direction, so a public name cannot
be added, removed or renamed without a reviewed change to the snapshot.

``__all__`` is read from the syntax tree, not by importing, so the check needs
no environment. Only literal lists and tuples (plus ``+=`` of literals) are
understood; anything dynamic is an error rather than a silent omission.

Scope: private (underscore) modules are skipped, sample code under ``agent-samples/`` is excluded because the sample
migrations rewrite it on purpose, and ``tests/`` is not public API.

Usage::

    python3 .github/scripts/api_snapshot.py            # check
    python3 .github/scripts/api_snapshot.py --write    # regenerate api-snapshot.json
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "api-snapshot.json"
EXCLUDED_PREFIXES = ("agent-samples/", "tests/")
REGENERATE = "python3 .github/scripts/api_snapshot.py --write"


def _tracked_inits(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*__init__.py"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return sorted(p for p in out.splitlines() if not p.startswith(EXCLUDED_PREFIXES))


def _module_name(root: Path, rel: str) -> str:
    parts = list(Path(rel).parent.parts)
    start = len(parts)
    while start > 0 and (root / Path(*parts[:start]) / "__init__.py").exists():
        start -= 1
    # `start` is now the first directory of the unbroken package chain ending here.
    return ".".join(parts[start:])


def _literal_names(node: ast.expr, where: str) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise ValueError(f"{where}: __all__ must be a literal list or tuple")
    names = []
    for element in node.elts:
        if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
            raise ValueError(f"{where}: __all__ entries must be string literals")
        names.append(element.value)
    return names


def read_all(path: Path, where: str) -> list[str] | None:
    """Return the names in ``__all__``, or None when the module defines none."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=where)
    found: list[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            found = _literal_names(node.value, where)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            if node.value is not None:
                found = _literal_names(node.value, where)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            if found is None:
                raise ValueError(f"{where}: __all__ += before assignment")
            found += _literal_names(node.value, where)
    return found


def build(root: Path = ROOT) -> dict:
    modules: dict[str, dict[str, list[str]]] = {}
    for rel in _tracked_inits(root):
        names = read_all(root / rel, rel)
        if names is None:
            continue
        module = _module_name(root, rel)
        if any(part.startswith("_") for part in module.split(".")):
            continue  # private modules are not public API
        if module in modules:
            raise ValueError(f"{rel}: module {module} already recorded from another path")
        if len(set(names)) != len(names):
            raise ValueError(f"{rel}: duplicate names in __all__")
        modules[module] = {"path": rel, "all": sorted(names)}
    return {"version": 1, "modules": dict(sorted(modules.items()))}


def _render(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=False) + "\n"


def _diff(recorded: dict, actual: dict) -> list[str]:
    lines: list[str] = []
    old, new = recorded.get("modules", {}), actual["modules"]
    for module in sorted(old.keys() | new.keys()):
        if module not in new:
            lines.append(f"- module removed: {module}")
        elif module not in old:
            lines.append(f"+ module added: {module} ({len(new[module]['all'])} names)")
        else:
            before, after = set(old[module]["all"]), set(new[module]["all"])
            lines += [f"+ {module}: {name}" for name in sorted(after - before)]
            lines += [f"- {module}: {name}" for name in sorted(before - after)]
            if old[module].get("path") != new[module]["path"]:
                lines.append(f"~ {module}: path {old[module].get('path')} -> {new[module]['path']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="regenerate api-snapshot.json")
    parser.add_argument("--check", action="store_true", help="verify the snapshot (the default)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (for tests)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    snapshot_path = root / "api-snapshot.json"
    actual = build(root)
    if args.write:
        snapshot_path.write_text(_render(actual), encoding="utf-8")
        print(f"wrote {snapshot_path.name}: {len(actual['modules'])} modules")
        return 0
    if not snapshot_path.exists():
        print(f"api-snapshot.json is missing; run: {REGENERATE}", file=sys.stderr)
        return 1
    recorded = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if recorded == actual:
        print(f"api snapshot matches ({len(actual['modules'])} modules)")
        return 0
    print("Public API (__all__) changed without an updated snapshot:", file=sys.stderr)
    for line in _diff(recorded, actual) or ["~ snapshot metadata differs"]:
        print(f"  {line}", file=sys.stderr)
    print(f"If the change is intended, run: {REGENERATE} and review the diff.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
