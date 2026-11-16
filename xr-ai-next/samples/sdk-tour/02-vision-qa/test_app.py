# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Testing an app: fake models, a scripted session, and the trace.

Nothing here starts a model or a device hub. `FakeModels` answers each role
from a script; `ToolCall` scripts the language model asking for a tool. The
session records everything said, and the TRACE records every step with the
turn it belongs to, so a test can check WHY something was said, not only
what.

Run from this directory: uv --config-file ../../../../uv.toml run pytest
"""

import asyncio
import importlib

import app as module
import pytest
from xr_ai_next.testing import FakeModels, ToolCall, solid_frame


@pytest.fixture
def sample():
    """A fresh app for each test: an app starts once, and a test session
    starts and stops it."""
    return importlib.reload(module)


async def until(check, timeout_s: float = 5.0) -> None:
    """Wait for `check()`: a tool that waits on the camera finishes after
    `settle()` returns (settle waits only for model calls)."""
    for _ in range(int(timeout_s / 0.02)):
        if check():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


@pytest.mark.asyncio
async def test_looks_before_answering(sample) -> None:
    models = FakeModels(
        {
            "answer_with_reasoning": [
                ToolCall("look", {"question": "what color is the mug?"}),
                "It's a red mug.",
            ],
            "perceive": ["A red mug on a desk."],
        }
    )
    async with sample.app.test_session(models=models) as s:
        await s.join()
        await s.show_frame(solid_frame("red"))
        await s.say("What color is my mug?")
        await s.settle()

        assert s.spoken() == ["It's a red mug."]

        # One turn; inside it the `look` tool, and inside the tool the
        # perception call (a span's event is recorded when the span ends).
        turn = s.trace.turns()[0]
        (look,) = s.trace.events(kind="tool", turn=turn)
        (seen,) = s.trace.calls("perceive")
        assert seen.parent == look.data["span"]
        assert seen.data["tool"] == "look"


@pytest.mark.asyncio
async def test_no_camera(sample) -> None:
    models = FakeModels(
        {
            "answer_with_reasoning": [
                ToolCall("look", {"question": "what is this?"}),
                "I can't see anything right now.",
            ],
        }
    )
    sample.SETTINGS.frame_timeout_s = 0.05  # no camera: give up quickly
    async with sample.app.test_session(models=models) as s:
        await s.join()
        await s.say("What is this?")
        await until(s.spoken)
        assert models.calls("perceive") == []
        result = s.trace.events(kind="tool")[0].data["result"]
        assert result == "The camera is not streaming."
