# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outputs: where what the assistant says goes (``p.say(..., to=sink)``, ``p.reply(..., to=sink)``).

A :class:`Sink` is any object with ``say``, ``reply`` and ``withdraw`` taking the participant first. ``Voice`` is the
usual one. :class:`TextSink` records text only (for typed-only apps and tests).
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ._base import _tracing
from ._base.participant import CURRENT_TURN
from .clock import wall_us

if TYPE_CHECKING:
    from ._base.participant import Participant

__all__ = ["Sink", "TextSink", "note_output", "output_started"]


@runtime_checkable
class Sink(Protocol):
    """Where speech goes. ``reply`` answers the current turn (streaming allowed); ``say`` speaks on the assistant's
    own initiative; ``withdraw`` drops a queued ``say`` tagged with ``source``."""

    @property
    def name(self) -> str: ...

    async def say(self, p: Participant, text: str, *, urgent: bool = False, source: str | None = None) -> None: ...

    async def reply(self, p: Participant, text: str | AsyncIterable[str]) -> str: ...

    async def withdraw(self, p: Participant, source: str) -> bool: ...


def note_output(sink: str, how: str, text: str) -> None:
    """Trace one output, and the turn's first output (``first_output``, with the delay since the turn's input)."""
    _tracing.emit("output", sink, how=how, text=text)
    turn = CURRENT_TURN.get()
    state = getattr(turn, "state", None)
    if turn is None or state is None or _tracing.current_ids().turn != turn.id:
        return
    if not state.get("_output_started"):
        state["_output_started"] = True
        _tracing.emit("first_output", sink, after_turn_s=round(wall_us() / 1e6 - turn.timestamp_us / 1e6, 3))


def output_started() -> bool:
    """Whether the current turn has said anything yet, on any sink (for example, to skip a "one moment" once the
    answer has started). ``False`` outside a turn."""
    state = getattr(CURRENT_TURN.get(), "state", None)
    return bool(state is not None and state.get("_output_started"))


async def mark_first(stream: AsyncIterable[str], on_first: object) -> AsyncIterator[str]:
    """Pass ``stream`` through, calling ``on_first()`` before the first non-empty fragment."""
    first = True
    async for part in stream:
        if first and part:
            first = False
            on_first()  # type: ignore[operator]
        yield part


class TextSink:
    """A sink that only records what would be said (per participant), for typed apps and tests."""

    def __init__(self, name: str = "text") -> None:
        self._name = name
        self.said: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return self._name

    async def say(self, p: Participant, text: str, *, urgent: bool = False, source: str | None = None) -> None:
        note_output(self._name, "say", text)
        self.said.setdefault(p.id, []).append(text)

    async def reply(self, p: Participant, text: str | AsyncIterable[str]) -> str:
        whole = text if isinstance(text, str) else "".join([part async for part in text])
        note_output(self._name, "reply", whole)
        self.said.setdefault(p.id, []).append(whole)
        return whole

    async def withdraw(self, p: Participant, source: str) -> bool:
        return False
