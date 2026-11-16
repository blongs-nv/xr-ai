# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Testing ``xr_ai_next`` apps: the existing test kit plus the trace and a frame helper.

``app.test_session(models=FakeModels({...}))`` starts the app on a fake hub and scripted roles; the session's
``trace`` is a :class:`~xr_ai_next.trace.MemorySink` holding every event of the session.
"""

from __future__ import annotations

import io

from ._base._tracing import MemorySink
from ._base.frames import Frame
from ._base.testing import FakeCall, FakeHub, FakeModels, ScriptExhausted, TestSession, ToolCall, test_session
from .clock import wall_us

__all__ = [
    "FakeCall",
    "FakeHub",
    "FakeModels",
    "MemorySink",
    "ScriptExhausted",
    "TestSession",
    "ToolCall",
    "solid_frame",
    "test_session",
]


def solid_frame(color: str = "gray", *, width: int = 64, height: int = 48, timestamp_s: float | None = None) -> Frame:
    """A one-color JPEG frame stamped now (or at ``timestamp_s``), for tests that only need a camera image."""
    from PIL import Image  # noqa: PLC0415 - Pillow comes with the frame tools

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG")
    stamp = wall_us() / 1_000_000 if timestamp_s is None else timestamp_s
    return Frame(buffer.getvalue(), int(stamp * 1_000_000), width, height)


solid_frame.__test__ = False  # type: ignore[attr-defined]
