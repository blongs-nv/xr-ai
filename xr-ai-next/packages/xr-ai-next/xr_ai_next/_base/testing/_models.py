# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``FakeModels``: scripted model roles that record calls and honor cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .. import _tracing
from ..profiles import ROLE_CATEGORIES, SPEECH_CATEGORIES, Decision, Role, StructuredOutputError, speech_role_error

if TYPE_CHECKING:
    from ..frames import Frame

Reply = Any
"""A scripted reply: text, a ``BaseModel`` for ``response_format``, an exception to raise, or
a callable taking the :class:`FakeCall` and returning any of these."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A scripted reply in which the model asks for a tool instead of answering with text.

    ``FakeModels.script("answer", [ToolCall("save_note", {"text": "milk"}), "Saved."])`` makes the
    first model call request ``save_note`` and the second answer. ``ask`` returns the
    :class:`ToolCall` itself and a stream of it yields only an empty chunk; the conversation's tool loop reads
    it from the call (:attr:`FakeCall.tool_call`) and runs the tool.
    """

    name: str
    """Name of the tool the model calls."""

    arguments: Mapping[str, Any] = field(default_factory=dict)
    """Arguments the model passes, as the parsed JSON object."""


class ScriptExhausted(LookupError):
    """A role was called more often than its script allows."""


@dataclass(slots=True)
class FakeCall:
    """One call a fake role received."""

    role: str
    method: str
    """``ask``, ``stream`` or ``decide``."""

    prompt: str
    frame: Frame | None = None
    """The frame of the call; the first one when several were passed (see ``frames``)."""

    response_format: type[BaseModel] | None = None
    choices: tuple[str, ...] = ()
    context: str = ""
    frames: tuple[Frame, ...] = ()
    """Every frame the call received, in order."""

    tool_call: ToolCall | None = None
    """Set when the scripted reply to this call was a :class:`ToolCall`."""

    criteria: Mapping[str, str] = field(default_factory=dict)
    """``decide`` only: the criteria text per choice when ``choices`` was a mapping."""

    system: str | None = None
    thinking: bool | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    timeout_s: float | None = None
    started: bool = False
    finished: bool = False
    """The call returned or its stream ran to the end."""

    cancelled: bool = False
    """The call was cancelled or its stream was closed before the end."""

    close_count: int = 0
    """Times the stream was closed (R5.3 expects exactly one for a cancelled stream)."""


class _RoleState:
    def __init__(self, replies: Sequence[Reply]) -> None:
        self.replies = list(replies)
        self.cursor = 0
        self.latency_s = 0.0
        self.chunk_latency_s = 0.0
        self.calls: list[FakeCall] = []
        self.started = asyncio.Event()
        self.started_count = 0


class FakeModels:
    """Scripted model roles. Each role replays its replies in order and records every call.

    ``script`` maps a role name (``answer``, ``decide``, ...) to its replies. A reply is a string,
    a ``BaseModel`` (for ``ask(..., response_format=...)``), an exception instance (raised), or a
    callable taking the :class:`FakeCall`. A role whose script runs out raises
    :class:`ScriptExhausted`, so an unexpected extra model call fails the test.

    Latency is set per role with :meth:`set_latency`. A call waits that long before it answers,
    and each streamed chunk waits ``chunk_s``; both are ``asyncio.sleep`` and so honor
    cancellation. Cancelling a call, or closing its stream, is recorded on the :class:`FakeCall`.
    """

    def __init__(self, script: Mapping[str, Sequence[Any]] | None = None) -> None:
        self._roles: dict[str, _RoleState] = {}
        for name, replies in (script or {}).items():
            self._roles[name] = _RoleState(replies)

    def script(self, role: str, replies: Sequence[Any]) -> None:
        """Replace ``role``'s replies and restart it from the first one."""
        state = self._state(role)
        state.replies = list(replies)
        state.cursor = 0

    def set_latency(self, role: str, seconds: float, *, chunk_s: float = 0.0) -> None:
        """Make ``role`` wait ``seconds`` before answering and ``chunk_s`` before each streamed chunk."""
        state = self._state(role)
        state.latency_s = seconds
        state.chunk_latency_s = chunk_s

    def role(self, name: str) -> Role:
        """Return a :class:`~.profiles.Role` backed by this script.

        The name is used as given (no alias, no default name). The standard speech roles
        (``speech_in``, ``speech_out``) have no chat methods here either: ``ask``, ``stream`` and
        ``decide`` on them raise, as on a role bound to a speech entry.
        """
        self._state(name)
        return _FakeRole(self, name)

    def calls(self, role: str) -> list[Any]:
        """Return the calls the named role received."""
        return list(self._state(role).calls)

    def cancelled(self, role: str) -> int:
        """Number of calls to ``role`` that were cancelled or whose stream was closed early."""
        return sum(1 for c in self._state(role).calls if c.cancelled)

    def active(self, role: str) -> int:
        """Number of calls to ``role`` that started and have neither finished nor been cancelled."""
        return sum(1 for c in self._state(role).calls if c.started and not c.finished and not c.cancelled)

    def any_active(self) -> bool:
        """Whether any role has a call in flight."""
        return any(self.active(name) for name in self._roles)

    def pending_replies(self, role: str) -> int:
        """Scripted replies of ``role`` not yet consumed."""
        state = self._state(role)
        return len(state.replies) - state.cursor

    async def wait_started(self, role: str, count: int = 1, *, timeout_s: float = 5.0) -> None:
        """Wait until ``role`` has started ``count`` calls; raise ``TimeoutError`` otherwise."""
        state = self._state(role)

        async def _wait() -> None:
            while state.started_count < count:
                state.started.clear()
                await state.started.wait()

        await asyncio.wait_for(_wait(), timeout_s)

    # -- internals -------------------------------------------------------------------------

    def _state(self, role: str) -> _RoleState:
        state = self._roles.get(role)
        if state is None:
            state = self._roles[role] = _RoleState(())
        return state

    def _begin(self, role: str, call: FakeCall) -> _RoleState:
        state = self._state(role)
        call.started = True
        state.calls.append(call)
        state.started_count += 1
        state.started.set()
        return state

    def _next_reply(self, role: str, call: FakeCall) -> Any:
        state = self._state(role)
        if state.cursor >= len(state.replies):
            raise ScriptExhausted(f"role {role!r} has no scripted reply for call {len(state.calls)}")
        reply = state.replies[state.cursor]
        state.cursor += 1
        if callable(reply) and not isinstance(reply, (BaseModel, type)):
            reply = reply(call)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, ToolCall):
            call.tool_call = reply
        return reply


def _frame_fields(frame: Frame | Sequence[Frame] | None) -> dict[str, Any]:
    frames = () if frame is None else tuple(frame) if isinstance(frame, Sequence) else (frame,)
    return {"frame": frames[0] if frames else None, "frames": frames}


class _FakeRole(Role):
    def __init__(self, models: FakeModels, name: str) -> None:
        self._models = models
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _refuse_speech(self, method: str) -> None:
        category = next(iter(ROLE_CATEGORIES.get(self._name, ())), None)
        if category in SPEECH_CATEGORIES:
            raise speech_role_error(self._name, category, method)

    @_tracing.traced_call("ask")
    async def ask(
        self,
        frame: Frame | Sequence[Frame] | None,
        prompt: str,
        *,
        response_format: type[BaseModel] | None = None,
        system: str | None = None,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        self._refuse_speech("ask")
        call = FakeCall(
            self._name,
            "ask",
            prompt,
            response_format=response_format,
            system=system,
            thinking=thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            **_frame_fields(frame),
        )
        reply = await self._answer(call)
        if response_format is not None and not isinstance(reply, response_format):
            try:
                reply = response_format.model_validate(reply)
            except ValueError as exc:
                raise StructuredOutputError(f"reply is not a valid {response_format.__name__}: {exc}") from exc
        return reply

    @_tracing.traced_stream
    def stream(
        self,
        frame: Frame | Sequence[Frame] | None,
        prompt: str,
        *,
        system: str | None = None,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[str]:
        self._refuse_speech("stream")
        call = FakeCall(
            self._name,
            "stream",
            prompt,
            system=system,
            thinking=thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            **_frame_fields(frame),
        )
        return _FakeStream(self._models, call)

    @_tracing.traced_call("decide")
    async def decide(
        self,
        question: str,
        *,
        choices: Sequence[str] | Mapping[str, str],
        context: str = "",
        frames: Sequence[Frame] = (),
    ) -> Decision:
        self._refuse_speech("decide")
        call = FakeCall(
            self._name,
            "decide",
            question,
            choices=tuple(choices),
            context=context,
            criteria=dict(choices) if isinstance(choices, Mapping) else {},
            **_frame_fields(tuple(frames) if frames else None),
        )
        reply = await self._answer(call)
        label = str(reply)
        if label not in choices:
            raise ValueError(f"scripted decision {label!r} is not one of {list(choices)}")
        if isinstance(reply, Decision):
            return reply
        return Decision(label)

    async def _answer(self, call: FakeCall) -> Any:
        state = self._models._begin(self._name, call)
        try:
            if state.latency_s:
                if call.timeout_s is not None and state.latency_s > call.timeout_s:
                    await asyncio.sleep(call.timeout_s)
                    raise TimeoutError(f"role {self._name!r} call exceeded {call.timeout_s}s")
                await asyncio.sleep(state.latency_s)
            reply = self._models._next_reply(self._name, call)
        except asyncio.CancelledError:
            call.cancelled = True
            raise
        except Exception:
            call.finished = True  # a failed call is over, so ``active`` does not count it
            raise
        call.finished = True
        return reply


class _FakeStream:
    """Async iterator over a scripted reply, word by word; ``aclose`` is idempotent and counted once."""

    def __init__(self, models: FakeModels, call: FakeCall) -> None:
        self._models = models
        self._call = call
        self._chunks: list[str] | None = None
        self._next = 0
        self._closed = False

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> str:
        if self._closed:
            raise StopAsyncIteration
        call = self._call
        state = self._models._state(call.role)
        try:
            if self._chunks is None:
                state = self._models._begin(call.role, call)
                if state.latency_s:
                    await asyncio.sleep(state.latency_s)
                reply = self._models._next_reply(call.role, call)
                text = "" if isinstance(reply, ToolCall) else reply if isinstance(reply, str) else str(reply)
                words = text.split(" ")
                self._chunks = [w + " " for w in words[:-1]] + [words[-1]]
            elif state.chunk_latency_s:
                await asyncio.sleep(state.chunk_latency_s)
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception:
            self._closed = True
            call.finished = True
            raise
        if self._next >= len(self._chunks):
            self._closed = True
            call.finished = True
            raise StopAsyncIteration
        chunk = self._chunks[self._next]
        self._next += 1
        return chunk

    async def aclose(self) -> None:
        """Close the stream; the first close of an unfinished stream counts as a cancellation."""
        if self._closed:
            return
        self._closed = True
        self._call.close_count += 1
        if not self._call.finished:
            self._call.cancelled = True
