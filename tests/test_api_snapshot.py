# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``__all__`` snapshot passes on this tree and fails on an unreviewed change."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / ".github" / "scripts" / "api_snapshot.py"
_spec = importlib.util.spec_from_file_location("api_snapshot", _SCRIPT)
api_snapshot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api_snapshot)


def test_checked_in_snapshot_matches_the_tree(capsys) -> None:
    assert api_snapshot.main([]) == 0, capsys.readouterr().err


def test_snapshot_covers_the_sdk_and_tools_surface() -> None:
    modules = json.loads((_ROOT / "api-snapshot.json").read_text())["modules"]
    assert {"xr_ai_next", "xr_ai_tools", "xr_ai_models", "xr_ai_runtime"} <= modules.keys()
    assert not any(m.startswith("agent") or "._" in m for m in modules)
    assert "App" in modules["xr_ai_next"]["all"]


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    pkg = tmp_path / "lib" / "demo_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('__all__ = ["alpha", "beta"]\n')
    (tmp_path / "agent-samples" / "s").mkdir(parents=True)
    (tmp_path / "agent-samples" / "s" / "__init__.py").write_text('__all__ = ["ignored"]\n')
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def test_unreviewed_change_fails_and_regeneration_passes(mini_repo: Path, capsys) -> None:
    root = ["--root", str(mini_repo)]
    assert api_snapshot.main([*root, "--write"]) == 0
    assert api_snapshot.main(root) == 0
    recorded = json.loads((mini_repo / "api-snapshot.json").read_text())
    assert list(recorded["modules"]) == ["demo_pkg"]  # sample packages are out of scope

    init = mini_repo / "lib" / "demo_pkg" / "__init__.py"
    init.write_text('__all__ = ["alpha", "beta", "gamma"]\n')
    assert api_snapshot.main(root) == 1
    err = capsys.readouterr().err
    assert "+ demo_pkg: gamma" in err and "--write" in err

    init.write_text('__all__ = ["alpha"]\n')
    assert api_snapshot.main(root) == 1
    assert "- demo_pkg: beta" in capsys.readouterr().err

    assert api_snapshot.main([*root, "--write"]) == 0
    assert api_snapshot.main(root) == 0


def test_dynamic_all_is_an_error_not_an_omission(mini_repo: Path) -> None:
    (mini_repo / "lib" / "demo_pkg" / "__init__.py").write_text("__all__ = sorted(['b', 'a'])\n")
    with pytest.raises(ValueError, match="literal"):
        api_snapshot.main(["--root", str(mini_repo), "--write"])
