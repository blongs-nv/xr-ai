# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Role limits and structured-output retry on bound roles (fake clients, no network)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from xr_ai_next._base import _bound
from xr_ai_next._base.app import App
from xr_ai_next._base.enrollment import PerConnection

_MODELS = {
    "models": {
        "llm": {"adapter": {"preset": "nemotron3_nano"}, "endpoint": {"base_url": "http://127.0.0.1:1"}},
        "vlm": {
            "category": "vlm",
            "adapter": {"kind": "openai_compat", "model_name": "v"},
            "endpoint": {"base_url": "http://127.0.0.1:2"},
        },
    },
    "roles": {"answer": "llm", "perceive": "vlm"},
    "budgets": {"answer": {"max_concurrency": 1}},
}


class FakeClient:
    def __init__(self) -> None:
        self.replies = ["hello"]
        self.calls = 0
        self.in_flight = 0
        self.peak = 0

    async def chat(self, messages: Any, **_: Any) -> Any:
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.02)
        finally:
            self.in_flight -= 1
        return SimpleNamespace(content=self.replies[min(self.calls, len(self.replies)) - 1])

    ask_image = chat

    async def close(self) -> None:
        pass


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[App, dict[str, FakeClient]]:
    (tmp_path / "models.json").write_text(json.dumps(_MODELS))
    (tmp_path / "app.yaml").write_text("{}")
    made: dict[str, FakeClient] = {}

    def maker(_m: Any, entry: str) -> FakeClient:
        made[entry] = FakeClient()
        return made[entry]

    for category in ("llm", "vlm"):
        monkeypatch.setitem(_bound._MAKERS, category, maker)
    return App.from_config(tmp_path / "app.yaml", enrollment=PerConnection(admit=None)), made


class Verdict(BaseModel):
    ok: bool


async def test_loop_and_foreground_turn_share_the_role_limit(app) -> None:
    app, clients = app
    role = app.role("answer")
    results = await asyncio.gather(role.ask(None, "loop tick"), role.ask(None, "foreground turn"), role.ask(None, "x"))
    assert results == ["hello"] * 3
    assert clients["llm"].peak == 1  # budgets.answer.max_concurrency: 1


async def test_structured_output_retries_once(app) -> None:
    app, clients = app
    role = app.role("answer")
    clients["llm"].replies = ["not json", '{"ok": true}']
    assert await role.ask(None, "q", response_format=Verdict) == Verdict(ok=True)
    assert clients["llm"].calls == 2
