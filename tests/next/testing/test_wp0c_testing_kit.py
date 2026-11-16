# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted tool calls and participant access in the test kit."""

from __future__ import annotations

from pathlib import Path

import pytest
from xr_ai_next._base.app import App, AppConfig
from xr_ai_next._base.testing import FakeModels, ToolCall


async def test_scripted_tool_call_is_returned_recorded_and_streams_no_text() -> None:
    models = FakeModels({"answer": [ToolCall("save_note", {"text": "milk"}), "Saved.", ToolCall("noop")]})
    role = models.role("answer")
    first = await role.ask(None, "remember milk")
    assert first == ToolCall("save_note", {"text": "milk"}) and first.arguments["text"] == "milk"
    assert await role.ask(None, "and?") == "Saved."
    assert [x async for x in role.stream(None, "again")] == [""]  # no text; the tool loop reads the call
    calls = models.calls("answer")
    assert [c.tool_call for c in calls] == [first, None, ToolCall("noop", {})]
    models.script("answer", [ToolCall("later")])  # script() accepts them too
    assert (await role.ask(None, "x")).name == "later"


async def test_session_participant_gives_the_scope_object_and_tick_settles_without_loops() -> None:
    app = App(AppConfig(config_dir=Path(".")))
    async with app.test_session() as s:
        with pytest.raises(KeyError, match="not joined"):
            s.participant()
        await s.join("p1")
        p = s.participant()
        assert p.id == "p1" and s.participant("p1") is p
        p.state.step = 2
        await s.settle()
        assert s.participant().state.step == 2
        await s.leave("p1")
        with pytest.raises(KeyError):
            s.participant("p1")
