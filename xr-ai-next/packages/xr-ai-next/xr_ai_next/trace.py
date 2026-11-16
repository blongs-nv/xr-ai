# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace events: what an app did, tied to what caused it.

Every event carries ``kind``, ``name``, the time and the ids of what caused it (session, participant, turn, task,
parent span), so a sink can tell which turn a model call or a tool belongs to.

Reading: ``app.trace`` is the app's :class:`Tracer`; add a sink with ``app.trace.add(sink)``. A sink is any callable
taking an :class:`Event`: :class:`MemorySink` (tests), :class:`JsonlSink`, :class:`LogSink`, :class:`RelaySink`
(NeMo Relay, so its ATOF/ATIF/OpenTelemetry exporters see the app).

Writing, for library and app code: :func:`emit` one event, :func:`span` a timed block (events inside it name it
as their parent), :func:`brief` a value for an event, :func:`in_tool` tag the model calls made inside a tool, and
:func:`quiet_role_calls` when code traces a model call itself (the role calls under it are then not traced twice).
All of them do nothing when no sink is listening.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ._base import _tracing
from ._base._relay import RelaySink
from ._base._tracing import Event, Ids, JsonlSink, LogSink, MemorySink, Sink, Tracer, current_ids

__all__ = [
    "Event",
    "Ids",
    "JsonlSink",
    "LogSink",
    "MemorySink",
    "RelaySink",
    "Sink",
    "Tracer",
    "brief",
    "current_ids",
    "emit",
    "in_tool",
    "quiet_role_calls",
    "span",
]


def emit(kind: str, name: str = "", **data: Any) -> None:
    """Record one event with the ids of the running code."""
    _tracing.emit(kind, name, **data)


@contextlib.asynccontextmanager
async def span(name: str, *, kind: str = "span", **data: Any) -> AsyncIterator[dict[str, Any]]:
    """Time a block; events inside it name it as their parent. Fill the yielded dict to add data to its event."""
    async with _tracing.span(name, kind=kind, **data) as record:
        yield record


def brief(value: Any, limit: int = 400) -> Any:
    """A short, JSON-friendly form of a value for an event (a model as its fields, long text cut)."""
    return _tracing._brief(value, limit)


@contextlib.contextmanager
def in_tool(name: str) -> Iterator[None]:
    """Model calls made inside the block are recorded as made by tool ``name``."""
    token = _tracing.CURRENT_TOOL.set(name)
    try:
        yield
    finally:
        _tracing.CURRENT_TOOL.reset(token)


@contextlib.contextmanager
def quiet_role_calls() -> Iterator[None]:
    """Role calls inside the block are not traced (the code around them records the call itself)."""
    token = _tracing.IN_CHAT.set(True)
    try:
        yield
    finally:
        _tracing.IN_CHAT.reset(token)
