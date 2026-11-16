# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace events (private core; the public face is :mod:`xr_ai_app.trace`): what an app did, and why.

Every event carries the ids of its cause: the ``session`` (connection), the ``participant``, the ``turn`` that
started it, the ``task`` (named background work) it ran in and its ``parent`` span. The ids flow through
:mod:`contextvars`, so a line spoken by a background task long after the turn ended still names that turn.

The app emits events by itself (``turn``, ``route``, ``role``, ``tool``, ``context``, ``output``,
``first_output``, ``spawn``, ``cancel``, ``job``, ``span``, ``error``). Code adds its own with
:meth:`Tracer.emit` and :meth:`Tracer.span`. Where events go is a list of sinks: any callable taking an
:class:`Event`. A sink must not block; a sink that raises is reported once and then skipped for that error.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import itertools
import json
import time
from collections.abc import Callable, Coroutine, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger

__all__ = [
    "Event",
    "JsonlSink",
    "LogSink",
    "MemorySink",
    "Sink",
    "Tracer",
    "current_ids",
]

Sink = Callable[["Event"], None]
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

_ids = itertools.count(1)


def new_id(prefix: str) -> str:
    """A short id, unique in this process (``t12``, ``s3``...)."""
    return f"{prefix}{next(_ids)}"


@dataclass(frozen=True, slots=True)
class Ids:
    """The cause of whatever runs now."""

    session: str | None = None
    participant: str | None = None
    turn: str | None = None
    task: str | None = None
    parent: str | None = None


_CURRENT: contextvars.ContextVar[Ids] = contextvars.ContextVar("xr_ai_trace_ids", default=Ids())  # noqa: B039
_TRACER: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar("xr_ai_tracer", default=None)


def current_ids() -> Ids:
    """The ids of the running code (all ``None`` outside an app)."""
    return _CURRENT.get()


@contextlib.contextmanager
def bound(**ids: str | None) -> Iterator[Ids]:
    """Run the block with some ids replaced (the others are inherited)."""
    now = _CURRENT.get()
    values = {name: getattr(now, name) for name in Ids.__slots__}
    values.update(ids)
    new = Ids(**values)
    token = _CURRENT.set(new)
    try:
        yield new
    finally:
        _CURRENT.reset(token)


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened."""

    kind: str
    """What kind of thing: ``turn``, ``route``, ``role``, ``tool``, ``context``, ``output``, ``span``..."""
    name: str
    """Which one: the role name, tool name, channel name, destination name, span name."""
    id: str
    t: float
    """Wall-clock time (``time.time()``) when it started."""
    session: str | None = None
    participant: str | None = None
    turn: str | None = None
    task: str | None = None
    parent: str | None = None
    duration_s: float | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, ensure_ascii=False)


class Tracer:
    """The app's trace: a list of sinks, and the calls that emit to them (``app.trace``)."""

    def __init__(self) -> None:
        self._sinks: list[Sink] = []
        self._failed: set[tuple[int, type]] = set()

    def add(self, sink: Sink) -> Sink:
        """Send every event to ``sink`` from now on; returns it (usable as a decorator)."""
        self._sinks.append(sink)
        return sink

    def remove(self, sink: Sink) -> None:
        self._sinks.remove(sink)

    @property
    def active(self) -> bool:
        return bool(self._sinks)

    def emit(
        self, kind: str, name: str = "", *, duration_s: float | None = None, t: float | None = None, **data: Any
    ) -> Event | None:
        """Record one event, with the ids of the running code. Returns it (``None`` without sinks)."""
        if not self._sinks:
            return None
        ids = _CURRENT.get()
        event = Event(
            kind=kind,
            name=name,
            id=new_id("e"),
            t=time.time() if t is None else t,
            session=ids.session,
            participant=ids.participant,
            turn=ids.turn,
            task=ids.task,
            parent=ids.parent,
            duration_s=duration_s,
            data=data,
        )
        for sink in list(self._sinks):
            try:
                sink(event)
            except Exception as exc:  # noqa: BLE001 - a sink must never break the app
                key = (id(sink), type(exc))
                if key not in self._failed:
                    self._failed.add(key)
                    logger.opt(exception=exc).warning("trace sink {!r} failed", sink)
        return event

    @contextlib.asynccontextmanager
    async def span(self, name: str, *, kind: str = "span", **data: Any):  # noqa: ANN201 - async context manager
        """Time a block; events inside it name this span as their ``parent``.

        The span's own event is emitted when the block ends, with ``duration_s`` and, on failure,
        ``error``. The yielded dict may be filled with more data while the block runs.
        """
        span_id = new_id("sp")
        start_wall, start = time.time(), time.monotonic()
        extra: dict[str, Any] = {}
        error: str | None = None
        with bound(parent=span_id):
            try:
                yield extra
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                self._emit_span(kind, name, span_id, start_wall, time.monotonic() - start, data, extra, error)

    def _emit_span(
        self,
        kind: str,
        name: str,
        span_id: str,
        t: float,
        duration_s: float,
        data: Mapping[str, Any],
        extra: Mapping[str, Any],
        error: str | None,
    ) -> None:
        if not self._sinks:
            return
        merged = {**data, **extra, "span": span_id}
        if error is not None:
            merged["error"] = error
        self.emit(kind, name, duration_s=duration_s, t=t, **merged)


# ---------------------------------------------------------------- the running app's tracer


_GLOBAL: list[Tracer] = []


def tracer() -> Tracer | None:
    """The tracer of the app this code runs in: the context's, else the last app started (``None`` outside apps)."""
    t = _TRACER.get()
    if t is None and _GLOBAL:
        return _GLOBAL[-1]
    return t


def emit(kind: str, name: str = "", **data: Any) -> None:
    """Emit through the running app's tracer, if any (internal convenience)."""
    t = tracer()
    if t is not None and t.active:
        t.emit(kind, name, **data)


async def in_task(coro: Coroutine[Any, Any, T], key: str | None) -> T:
    """Run ``coro`` as named background work: its events carry a new ``task`` id (and the turn that started it)."""
    task_id = new_id("k")
    name = key or getattr(coro, "__qualname__", "task")
    emit("spawn", name, task_id=task_id)
    started = time.monotonic()
    outcome = "done"
    with bound(task=task_id):
        try:
            return await coro
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException as exc:
            outcome = f"failed: {type(exc).__name__}"
            raise
        finally:
            emit("task_end", name, task_id=task_id, outcome=outcome, after_s=round(time.monotonic() - started, 3))


@contextlib.asynccontextmanager
async def span(name: str, *, kind: str = "span", **data: Any):  # noqa: ANN201
    t = tracer()
    if t is None or not t.active:
        yield {}
        return
    async with t.span(name, kind=kind, **data) as extra:
        yield extra


# ---------------------------------------------------------------- sinks


class MemorySink:
    """Keeps every event in a list, with a few queries; the test kit's ``session.trace`` is one."""

    def __init__(self) -> None:
        self.all: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.all.append(event)

    def events(self, *, kind: str | None = None, name: str | None = None, turn: str | None = None) -> list[Event]:
        """Events in order, filtered by kind, name and turn."""
        return [
            e
            for e in self.all
            if (kind is None or e.kind == kind)
            and (name is None or e.name == name)
            and (turn is None or e.turn == turn)
        ]

    def turns(self) -> list[str]:
        """Turn ids in the order the turns started."""
        return [e.data["id"] for e in self.all if e.kind == "turn" and e.data.get("phase") == "start"]

    def calls(self, role: str) -> list[Event]:
        """The model calls made through ``role``."""
        return self.events(kind="role", name=role)

    def kinds(self, *, turn: str | None = None) -> list[str]:
        return [e.kind for e in self.events(turn=turn)]

    def clear(self) -> None:
        self.all.clear()


class JsonlSink:
    """Appends one JSON line per event to ``<directory>/trace-<date>.jsonl``."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: Event) -> None:
        day = time.strftime("%Y-%m-%d", time.localtime(event.t))
        with (self._dir / f"trace-{day}.jsonl").open("a", encoding="utf-8") as file:
            file.write(event.to_json() + "\n")


class LogSink:
    """Logs each event on one line through loguru (``level`` defaults to DEBUG)."""

    def __init__(self, level: str = "DEBUG") -> None:
        self._level = level

    def __call__(self, event: Event) -> None:
        took = f" {event.duration_s * 1000:.0f}ms" if event.duration_s is not None else ""
        logger.log(
            self._level,
            "trace {} {}{} turn={} task={} {}",
            event.kind,
            event.name,
            took,
            event.turn,
            event.task,
            dict(event.data),
        )


# ---------------------------------------------------------------- model and tool calls

CURRENT_TOOL: contextvars.ContextVar[str | None] = contextvars.ContextVar("xr_ai_current_tool", default=None)
IN_CHAT: contextvars.ContextVar[bool] = contextvars.ContextVar("xr_ai_in_chat", default=False)
"""True inside a chat round that traces itself (``xr_ai_kit.Conversation``): role calls under it are not
traced again."""
"""The tool running now (set by the ``xr_ai_app`` tool loop); recorded on the model calls it makes."""


def _brief(value: Any, limit: int = 400) -> Any:
    """A short, JSON-friendly form of a model answer for the trace."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump()
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def traced_call(method: str) -> Callable[[F], F]:
    """Decorate a role's async ``ask``/``decide``: one ``role`` event per call (a no-op without sinks)."""

    def decorate(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            t = tracer()
            if t is None or not t.active or IN_CHAT.get():
                return await fn(self, *args, **kwargs)
            started_wall, started = time.time(), time.monotonic()
            data: dict[str, Any] = {"method": method}
            prompt = kwargs.get("prompt", kwargs.get("question"))
            if prompt is None and args:
                prompt = args[1] if method == "ask" and len(args) > 1 else args[0] if method == "decide" else None
            if isinstance(prompt, str):
                data["prompt"] = _brief(prompt)
            if (tool := CURRENT_TOOL.get()) is not None:
                data["tool"] = tool
            if "response_format" in kwargs and kwargs["response_format"] is not None:
                data["schema"] = kwargs["response_format"].__name__
            try:
                result = await fn(self, *args, **kwargs)
            except BaseException as exc:
                data["error"] = type(exc).__name__
                took = time.monotonic() - started
                t.emit("role", getattr(self, "name", "?"), duration_s=took, t=started_wall, **data)
                raise
            data["answer"] = _brief(result)
            if (confidence := getattr(result, "confidence", None)) is not None:
                data["confidence"] = confidence
            t.emit("role", getattr(self, "name", "?"), duration_s=time.monotonic() - started, t=started_wall, **data)
            return result

        return wrapper  # type: ignore[return-value]

    return decorate


def traced_stream(fn: F) -> F:
    """Decorate a role's ``stream``: one ``role`` event when the stream ends (a no-op without sinks)."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        stream = fn(self, *args, **kwargs)
        t = tracer()
        if t is None or not t.active or IN_CHAT.get():
            return stream
        return _TracedStream(t, getattr(self, "name", "?"), stream)

    return wrapper  # type: ignore[return-value]


class _TracedStream:
    def __init__(self, t: Tracer, name: str, inner: Any) -> None:
        self._t, self._name, self._inner = t, name, inner
        self._parts: list[str] = []
        self._wall, self._start = time.time(), time.monotonic()
        self._tool = CURRENT_TOOL.get()
        self._done = False

    def __aiter__(self) -> _TracedStream:
        return self

    async def __anext__(self) -> str:
        try:
            part = await self._inner.__anext__()
        except StopAsyncIteration:
            self._finish(None)
            raise
        except BaseException as exc:
            self._finish(type(exc).__name__)
            raise
        self._parts.append(part)
        return part

    def _finish(self, error: str | None) -> None:
        if self._done:
            return
        self._done = True
        data: dict[str, Any] = {"method": "stream", "answer": _brief("".join(self._parts))}
        if self._tool is not None:
            data["tool"] = self._tool
        if error is not None:
            data["error"] = error
        self._t.emit("role", self._name, duration_s=time.monotonic() - self._start, t=self._wall, **data)

    async def aclose(self) -> None:
        self._finish("closed")
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def activate(t: Tracer) -> None:
    """Make ``t`` the running app's tracer (the app calls it at start).

    Hub and runtime callbacks do not always run in the starting task's context, so the last started app's tracer
    is also the process-wide fallback.
    """
    _TRACER.set(t)
    if t in _GLOBAL:
        _GLOBAL.remove(t)
    _GLOBAL.append(t)


def deactivate(t: Tracer) -> None:
    """Forget ``t`` (the app calls it at stop)."""
    if t in _GLOBAL:
        _GLOBAL.remove(t)
