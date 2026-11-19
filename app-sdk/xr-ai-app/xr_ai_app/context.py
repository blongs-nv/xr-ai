# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context channels: the text components share with models, written and read explicitly.

A channel is a plain declaration (it is not added to the app). Its text lives in the state of the scope it
names, so it is cleaned up with that state:

==============  ======================================  =========================
scope           stored in                               lives as long as
==============  ======================================  =========================
``app``         the channel itself                      the app
``participant`` ``p.state``                             the participant
``session``     ``p.session.state``                     one connection
==============  ======================================  =========================

Two kinds. :class:`ContextValue` holds one text that each ``set`` replaces. :class:`ContextLog` appends entries,
keeps recent ones word for word and, when it outgrows its budget, folds the oldest into a rolling summary with
its ``summarize`` role, in the background (``publish`` never waits). Without a ``summarize`` role the oldest
entries are dropped instead.

A model sees exactly the channels a component lists (``xr_ai_kit.Conversation(context=[...])``); :func:`render`
is how. A component that reacts to new text (an escalator that learns from the dialogue) follows
:func:`changes`: ``async for _ in dialogue.changes(p):``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ._base import _tracing as trace

if TYPE_CHECKING:
    from ._base.participant import Participant
    from ._base.profiles import Role

__all__ = ["ContextChannel", "ContextLog", "ContextValue", "Scope", "render"]

Scope = Literal["app", "participant", "session"]
_SCOPES = ("app", "participant", "session")

_SUMMARY_PROMPT = (
    "Fold these notes into the running summary. Keep names, numbers, decisions and anything the user asked "
    "for; drop chit-chat. Answer with the new summary only, at most {limit} characters."
)


def clip(text: str, budget_bytes: int, *, keep: Literal["start", "end"] = "start") -> str:
    """``text`` cut to at most ``budget_bytes`` UTF-8 bytes, on a character boundary, marked with an ellipsis."""
    data = text.encode()
    if len(data) <= budget_bytes:
        return text
    room = max(budget_bytes - 3, 0)
    if keep == "start":
        return data[:room].decode(errors="ignore") + "..."
    return "..." + data[len(data) - room :].decode(errors="ignore")


@runtime_checkable
class ContextChannel(Protocol):
    """Anything that renders text for a participant: a channel, or a component's own (``voice.dialogue``)."""

    @property
    def name(self) -> str: ...

    async def view(self, p: Participant | None, *, budget_bytes: int | None = None) -> str: ...


class _Channel:
    def __init__(self, name: str, *, scope: Scope, budget_bytes: int) -> None:
        if scope not in _SCOPES:
            raise ValueError(f"context channel {name!r}: scope must be one of {_SCOPES}, got {scope!r}")
        if budget_bytes <= 0:
            raise ValueError(f"context channel {name!r}: budget_bytes must be positive")
        self._name = name
        self._scope: Scope = scope
        self._budget = budget_bytes
        self._app_store: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def scope(self) -> Scope:
        return self._scope

    @property
    def budget_bytes(self) -> int:
        return self._budget

    def _store(self, p: Participant | None) -> dict[str, Any]:
        if self._scope == "app":
            return self._app_store
        if p is None:
            raise ValueError(f"context channel {self._name!r} has {self._scope} scope: pass the participant")
        state = p.state if self._scope == "participant" else p.session.state
        key = f"_context:{self._name}"
        store = state.get(key)
        if store is None:
            store = state[key] = {}
        return store

    def _budget_for(self, budget_bytes: int | None) -> int:
        return self._budget if budget_bytes is None else min(budget_bytes, self._budget)

    def changes(self, p: Participant | None) -> AsyncIterator[None]:
        """This channel's :func:`changes`: ``async for _ in channel.changes(p):`` runs now, then after each
        ``set``, ``publish`` or ``clear``."""
        return changes(p, [self])

    def _watchers(self, p: Participant | None) -> set[asyncio.Event]:
        return self._store(p).setdefault("watchers", set())

    def _notify(self, p: Participant | None) -> None:
        for watcher in self._watchers(p):
            watcher.set()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name!r}, scope={self._scope!r})"


class ContextValue(_Channel):
    """One text, replaced by every :meth:`set`; never summarized."""

    async def set(self, p: Participant | None, text: str) -> None:
        """Replace the text (cut to the budget). ``p`` is ``None`` only for app scope."""
        text = clip(text.strip(), self._budget)
        self._store(p)["text"] = text
        trace.emit("context", self._name, op="set", bytes=len(text.encode()))
        self._notify(p)

    async def clear(self, p: Participant | None) -> None:
        self._store(p).pop("text", None)
        trace.emit("context", self._name, op="clear")
        self._notify(p)

    async def view(self, p: Participant | None, *, budget_bytes: int | None = None) -> str:
        """The text, at most ``budget_bytes`` (never more than the channel's budget); ``""`` when unset."""
        return clip(self._store(p).get("text", ""), self._budget_for(budget_bytes))

    def get(self, p: Participant | None) -> str:
        """The text, synchronously."""
        return self._store(p).get("text", "")


@dataclass
class _Log:
    entries: list[str] = field(default_factory=list)
    summary: str = ""
    dropped: int = 0
    folding: asyncio.Task[None] | None = None


class ContextLog(_Channel):
    """Entries appended by :meth:`publish`; the oldest are summarized (or dropped) to stay within budget."""

    def __init__(self, name: str, *, scope: Scope, budget_bytes: int, summarize: Role | None = None) -> None:
        super().__init__(name, scope=scope, budget_bytes=budget_bytes)
        self._summarize = summarize

    def _log(self, p: Participant | None) -> _Log:
        store = self._store(p)
        log = store.get("log")
        if log is None:
            log = store["log"] = _Log()
        return log

    async def publish(self, p: Participant | None, text: str) -> None:
        """Append one entry. Returns at once; summarizing happens in the background."""
        text = clip(text.strip(), self._budget // 2)
        if not text:
            return
        log = self._log(p)
        log.entries.append(text)
        trace.emit("context", self._name, op="publish", bytes=len(text.encode()))
        self._compact(log)
        self._notify(p)

    def entries(self, p: Participant | None) -> list[str]:
        """The entries kept word for word, oldest first (a copy)."""
        return list(self._log(p).entries)

    def summary(self, p: Participant | None) -> str:
        return self._log(p).summary

    async def clear(self, p: Participant | None) -> None:
        log = self._log(p)
        if log.folding is not None:
            log.folding.cancel()
        self._store(p)["log"] = _Log()
        trace.emit("context", self._name, op="clear")
        self._notify(p)

    async def view(self, p: Participant | None, *, budget_bytes: int | None = None) -> str:
        """The summary (if any), then the newest entries that fit in ``budget_bytes``."""
        log = self._log(p)
        budget = self._budget_for(budget_bytes)
        head = []
        if log.summary:
            head.append(f"(earlier, summarized) {log.summary}")
        elif log.dropped:
            head.append(f"({log.dropped} earlier entries dropped)")
        lines: list[str] = []
        used = sum(len(h.encode()) + 1 for h in head)
        for entry in reversed(log.entries):
            size = len(entry.encode()) + 1
            if used + size > budget:
                break
            lines.append(entry)
            used += size
        lines.reverse()
        return clip("\n".join([*head, *lines]), budget)

    async def settle(self, p: Participant | None) -> None:
        """Wait for a running summary to finish (tests and shutdown)."""
        folding = self._log(p).folding
        if folding is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await folding

    # ------------------------------------------------------------ compaction

    def _size(self, log: _Log) -> int:
        return len(log.summary.encode()) + sum(len(e.encode()) + 1 for e in log.entries)

    def _compact(self, log: _Log) -> None:
        if self._size(log) <= self._budget or log.folding is not None:
            return
        target = self._budget // 2
        old: list[str] = []
        while log.entries and self._size(log) > target:
            old.append(log.entries.pop(0))
        if not old:
            return
        if self._summarize is None:
            log.dropped += len(old)
            return
        log.folding = asyncio.get_running_loop().create_task(self._fold(log, old), name=f"context-fold:{self._name}")

    async def _fold(self, log: _Log, old: Sequence[str]) -> None:
        limit = max(self._budget // 4, 64)
        prompt = (f"Running summary:\n{log.summary}\n\n" if log.summary else "") + "New notes:\n" + "\n".join(old)
        try:
            assert self._summarize is not None
            text = await self._summarize.ask(
                None, prompt, system=_SUMMARY_PROMPT.format(limit=limit), thinking=False, max_tokens=256
            )
            log.summary = clip(str(text).strip(), limit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed summary drops the entries instead
            log.dropped += len(old)
            trace.emit("error", self._name, where="context summary", error=type(exc).__name__)
        finally:
            log.folding = None
        self._compact(log)


async def render(p: Participant | None, channels: Sequence[ContextChannel], *, budget_bytes: int | None = None) -> str:
    """The channels' text, each under its name, in the given order; empty channels are left out.

    ``budget_bytes`` caps each channel (never above its own budget).
    """
    parts: list[str] = []
    for channel in channels:
        text = await channel.view(p, budget_bytes=budget_bytes)
        if text.strip():
            parts.append(f"## {channel.name}\n{text.strip()}")
    return "\n\n".join(parts)


async def changes(p: Participant | None, channels: Sequence[ContextChannel]) -> AsyncIterator[None]:
    """Yield now, then after each change to any of ``channels`` for ``p``: ``async for _ in changes(p, [...]):``.

    Changes that come while the loop body runs are folded into one more turn, never lost. A channel that is not a
    :class:`ContextValue` or :class:`ContextLog` (a component's own view) yields no changes.
    """
    changed = asyncio.Event()
    watched = [channel for channel in channels if isinstance(channel, _Channel)]
    for channel in watched:
        channel._watchers(p).add(changed)
    try:
        while True:
            yield
            await changed.wait()
            changed.clear()
    finally:
        for channel in watched:
            channel._watchers(p).discard(changed)
