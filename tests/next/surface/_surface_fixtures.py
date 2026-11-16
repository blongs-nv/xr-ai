# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the surface tests: load the xr-ai-next samples (xr-ai-next/samples) as modules, behind a guard
profile, and guard against a Relay scope stack in the main context.

Registered as a plugin by ``tests/conftest.py`` (``pytest_plugins``), not a nested ``conftest.py``: a second
conftest would shadow ``tests/conftest.py`` for tests that ``import conftest`` (test_gpu_container_cleanup.py does).

Every sample builds its app and roles at import. The guard profile points every role at a closed port, so even a
role that escaped the test doubles could never reach a real model server.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

SAMPLES = Path(__file__).resolve().parents[3] / "xr-ai-next" / "samples" / "sdk-tour"


def _guard_profile(tmp: Path) -> Path:
    data = json.loads((SAMPLES / "models.json").read_text(encoding="utf-8"))

    def closed(value: object) -> object:
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                is_url = k == "base_url" and isinstance(v, str)
                out[k] = re.sub(r"//[^/]+", "//127.0.0.1:9", v, count=1) if is_url else closed(v)
            return out
        return value

    path = tmp / "guard_models.json"
    path.write_text(json.dumps(closed(data)), encoding="utf-8")
    return path


@pytest.fixture
def load_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], ModuleType]]:
    monkeypatch.setenv("XR_AI_PROFILE", str(_guard_profile(tmp_path)))
    added: list[str] = []

    def load(name: str) -> ModuleType:
        directory = SAMPLES / name
        sys.path.insert(0, str(directory))
        added.append(str(directory))
        module_name = f"sample_{name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, directory / "app.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield load
    for path in added:
        if path in sys.path:
            sys.path.remove(path)


_SURFACE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def no_relay_stack_in_the_main_context(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail a test that installs a NeMo Relay scope stack in pytest's main context.

    Relay keeps its scope stack in a context variable, and every asyncio task started from a context shares that
    context's stack object. A stack created here (any synchronous ``nemo_relay.scope`` call creates one) is
    inherited by every later test, which once made an unrelated voice pipeline test hang. Synchronous tests that
    use Relay run inside ``contextvars.copy_context().run``.
    """
    if _SURFACE not in Path(str(request.node.path)).resolve().parents:
        yield  # only the surface tests are held to this
        return
    import nemo_relay  # noqa: PLC0415 - optional at import time

    stack = nemo_relay._scope_stack_var  # private, but the one place the hazard is visible
    before = stack.get(None)
    yield
    assert stack.get(None) is before, "the test installed a Relay scope stack in the main context"
