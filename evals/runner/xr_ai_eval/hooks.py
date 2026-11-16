# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load ``path.py:function`` hooks (python scorers and call drivers) relative to a case file."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any


def split_spec(spec: Any) -> tuple[str, str]:
    """Split ``"file.py:function"``; raise ``ValueError`` when malformed."""
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"hook {spec!r} must look like 'file.py:function'")
    file_part, name = spec.rsplit(":", 1)
    if not file_part.endswith(".py") or not name.isidentifier():
        raise ValueError(f"hook {spec!r} must look like 'file.py:function'")
    return file_part, name


def resolve_hook_path(base_dir: Path, spec: str) -> Path:
    """Absolute path of the hook's file (relative to ``base_dir``, the case file's directory).

    The file must lie inside the suite directory (the parent of ``base_dir``); ``ValueError`` otherwise.
    """
    path = (base_dir / split_spec(spec)[0]).resolve()
    if not path.is_relative_to(base_dir.resolve().parent):
        raise ValueError(f"hook {spec!r} resolves outside the suite directory")
    return path


def load_hook(base_dir: Path, spec: str) -> Callable[..., Any]:
    """Import the hook's file by path (not through ``sys.path``) and return the function."""
    path = resolve_hook_path(base_dir, spec)
    name = split_spec(spec)[1]
    if not path.is_file():
        raise ValueError(f"hook file not found: {path}")
    module_name = "xr_ai_eval_hook_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"cannot import hook file {path}")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:  # a hook that fails to import is a case error, not a crash
        raise RuntimeError(f"hook file {path} failed to import: {type(exc).__name__}: {exc}") from exc
    function = getattr(module, name, None)
    if not callable(function):
        raise ValueError(f"hook {name!r} not found in {path}")
    return function


async def call_hook(function: Callable[..., Any], *args: Any) -> Any:
    """Call a sync or async hook; any failure becomes ``RuntimeError`` so the case records an error."""
    try:
        value = function(*args)
        if inspect.isawaitable(value):
            value = await value
        return value
    except Exception as exc:
        name = getattr(function, "__name__", function)
        raise RuntimeError(f"hook {name} failed: {type(exc).__name__}: {exc}") from exc
