# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Participant``, ``Session`` and ``Turn``: who is here, their connection's lifetime, and one user turn.

- ``p.state`` lives as long as the participant is enrolled; ``p.session.state`` starts empty on every (re)connect;
  ``turn.state`` is one turn's scratch state.
- ``p.say/reply/withdraw(..., to=sink)``: the output names where it goes (``Voice``, ``TextSink``...); without ``to``
  the app's speech output is used.
- ``p.spawn`` runs work in the participant's scope and carries the trace ids into the task.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterable, Coroutine, Iterator, MutableMapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger
from xr_ai_interaction._outputs import ResponseOutputs
from xr_ai_runtime import RuntimeClosedError

from . import _tracing
from .frames import FrameStream
from .scope import ParticipantScope
from .topics import OUTPUT

if TYPE_CHECKING:
    from xr_ai_runtime import AgentRuntime

    from ..output import Sink
    from .artifacts import ArtifactSink
    from .enrollment import Connection

T = TypeVar("T")

__all__ = ["CURRENT_TURN", "Participant", "ParticipantState", "Session", "Turn", "current_turn"]


def _closed_only(exc: BaseException) -> bool:
    """Whether ``exc`` (possibly a group from runtime delivery) only says the runtime or a delivery closed."""
    if isinstance(exc, BaseExceptionGroup):
        return all(_closed_only(inner) for inner in exc.exceptions)
    return isinstance(exc, (RuntimeClosedError, asyncio.CancelledError))


def _swallow_closed(exc: BaseException) -> bool:
    """Decide whether output code may drop ``exc``: closed-type errors, unless this task itself is cancelled."""
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        return False
    return _closed_only(exc)


class ParticipantState(MutableMapping[str, Any]):
    """State readable as a mapping and as attributes (``p.state``, ``p.session.state``, ``turn.state``)."""

    __slots__ = ("_data",)

    def __init__(self) -> None:
        object.__setattr__(self, "_data", {})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self._data[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self._data[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass(frozen=True, slots=True)
class Turn:
    """One user turn: ``text``, ``id``, ``timestamp_us``, and ``state`` (scratch state dropped when the turn ends)."""

    text: str
    """Gate-accepted speech or typed text."""

    timestamp_us: int
    """Input presentation time as Unix microseconds."""

    id: str
    """Identifier of this turn; cancelling the turn cancels its tool and model calls."""

    state: ParticipantState = field(default_factory=ParticipantState, compare=False, repr=False)
    """This turn's scratch state."""

    def replace(self, **changes: Any) -> Turn:
        """A copy with some fields changed; it shares this turn's ``state``."""
        changes.setdefault("state", self.state)
        return dataclasses.replace(self, **changes)


CURRENT_TURN: ContextVar[Turn | None] = ContextVar("xr_ai_next_current_turn", default=None)
"""The turn the running task belongs to; set by the app around each turn, inherited by spawned tasks."""


def current_turn() -> Turn | None:
    """The turn the running code is answering, or ``None`` outside a turn (background work, a detached task)."""
    return CURRENT_TURN.get()


class Session:
    """One connection's lifetime: ``state`` starts empty on every connect."""

    __slots__ = ("id", "state")

    def __init__(self) -> None:
        self.id = _tracing.new_id("s")
        self.state = ParticipantState()

    def __repr__(self) -> str:
        return f"Session({self.id!r})"


def _sinks(to: Any) -> list[Any]:
    return list(to) if isinstance(to, (list, tuple)) else [to]


class Participant:
    """Everything an app knows about, and does for, one enrolled participant."""

    def __init__(self, participant_id: str) -> None:
        self._id = participant_id
        self._state = ParticipantState()
        self._session = Session()
        self._session_used = False
        self._scope = ParticipantScope(participant_id)
        self._runtime: AgentRuntime | None = None
        self._artifacts: ArtifactSink | None = None
        self._sourced: set[str] = set()
        self._frames = FrameStream()
        self._connections: list[Connection] = []
        """The live connections, oldest first; the last one is the current connection."""

    # ------------------------------------------------------------------ app side (private)

    def _bind(self, *, scope: ParticipantScope, runtime: AgentRuntime | None, artifacts: ArtifactSink | None) -> None:
        """Attach the app-owned scope, runtime and artifact sink (called by the app)."""
        self._scope = scope
        self._runtime = runtime
        self._artifacts = artifacts

    def _add_connection(self, connection: Connection) -> None:
        """A device connected: it becomes the current connection; a reconnect starts a new session."""
        if connection in self._connections:
            return
        if not self._connections and self._session_used:
            self._session = Session()
        self._session_used = True
        self._connections.append(connection)

    def _remove_connection(self, connection: Connection) -> bool:
        """A device disconnected; the frames of the participant are dropped when none is left."""
        if connection not in self._connections:
            return False
        self._connections.remove(connection)
        if not self._connections:
            self._frames._clear()
        return True

    async def _release(self) -> None:
        """Leave step 5: artifacts drain first, then frames and state are released."""
        try:
            self._frames._close()
            if self._artifacts is not None:
                await self._artifacts.drain()
        finally:
            self._state._data.clear()
            self._session.state.clear()
            self._sourced.clear()

    # ------------------------------------------------------------------ who

    @property
    def id(self) -> str:
        """The participant id: the client Identity field of its connections."""
        return self._id

    @property
    def name(self) -> str:
        """How to refer to this participant in text (the id, until the hub reports display names)."""
        return self._id

    @property
    def connected(self) -> bool:
        """True while this participant has at least one connection.

        An enrolled participant outlives its connections under ``Persistent`` enrollment; ``p.frames`` follows the
        current connection (the most recent one still connected) and ``latest()`` is ``None`` while disconnected.
        """
        return bool(self._connections)

    @property
    def session(self) -> Session:
        """The current connection's session; its ``state`` is dropped on disconnect."""
        return self._session

    @property
    def frames(self) -> FrameStream:
        """This participant's camera frames."""
        return self._frames

    @property
    def state(self) -> ParticipantState:
        """Scratch state that is dropped when the participant leaves."""
        return self._state

    @property
    def artifacts(self) -> ArtifactSink:
        """This participant's file writer, rooted at ``<artifacts_dir>/<id>`` (``artifacts_dir`` in ``app.yaml``)."""
        if self._artifacts is None:
            raise RuntimeError("no artifacts_dir is configured for this app")
        return self._artifacts

    # ------------------------------------------------------------------ output

    async def say(
        self, text: str, *, urgent: bool = False, source: str | None = None, to: Sink | Sequence[Sink] | None = None
    ) -> None:
        """Speak one complete message on the assistant's own initiative.

        ``to`` names the output (a sink such as ``Voice`` or ``TextSink``, or a list of them); ``to=None`` speaks on
        the app's speech output (the voice pipeline ``app.yaml`` configures; a test session's hub). ``urgent``
        interrupts lower-priority output. ``source`` tags the utterance: a queued, not yet spoken utterance with the
        same ``source`` is replaced by this one (latest wins), and :meth:`withdraw` removes it. A silent no-op once
        the participant's scope stopped admitting or the runtime closed (rule R6.1).
        """
        if to is None:
            await self._say_default(text, urgent=urgent, source=source)
            return
        for sink in _sinks(to):
            await sink.say(self, text, urgent=urgent, source=source)

    async def reply(self, text: str | AsyncIterable[str], *, to: Sink | Sequence[Sink] | None = None) -> str:
        """Answer the current turn on ``to`` (streaming allowed); returns what was spoken.

        ``to=None`` answers on the app's speech output, as a reply to the current turn (outside a turn, as
        :meth:`say`). Returns ``""`` when nothing was spoken because the scope stopped admitting.
        """
        if to is None:
            return await self._reply_default(text)
        sinks = _sinks(to)
        if len(sinks) == 1:
            return await sinks[0].reply(self, text)
        whole = text if isinstance(text, str) else "".join([part async for part in text])
        for sink in sinks:
            await sink.reply(self, whole)
        return whole

    async def withdraw(self, source: str, *, to: Sink | Sequence[Sink] | None = None) -> bool:
        """Drop a queued ``say`` tagged ``source``; returns whether one was still tracked.

        ``to=None`` withdraws from the app's speech output: best effort, because the voice pipeline does not report
        playback, so an utterance that already played still counts until it is withdrawn, replaced or the
        participant leaves. Speech that started playing is never cut.
        """
        if to is None:
            return await self._withdraw_default(source)
        return any([await sink.withdraw(self, source) for sink in _sinks(to)])

    def _outputs(self, source: str | None = None) -> ResponseOutputs | None:
        if self._runtime is None or not self._scope.admitting:
            return None
        # Each ``source`` is its own producer, so the consumer's per-producer ``withdraw`` removes
        # exactly that source's queued utterance and nothing else.
        producer = "app" if source is None else f"app:{source}"
        return ResponseOutputs(self._runtime, OUTPUT, source=producer, closed_errors=(RuntimeClosedError,))

    async def _say_default(self, text: str, *, urgent: bool = False, source: str | None = None) -> None:
        """``say`` on the app's speech output (the ``OUTPUT`` topic, bridged to the voice pipeline or a fake hub)."""
        outputs = self._outputs(source)
        if outputs is None:
            return
        try:
            if source is not None:
                await outputs.withdraw(self._id)
                self._sourced.add(source)
            await outputs.update(self._id, text, urgent=urgent)
        except (RuntimeClosedError, BaseExceptionGroup) as exc:
            if not _swallow_closed(exc):
                raise
            logger.debug("say dropped for {}: runtime closed", self._id)

    async def _withdraw_default(self, source: str) -> bool:
        outputs = self._outputs(source)
        if outputs is None or source not in self._sourced:
            return False
        self._sourced.discard(source)
        try:
            await outputs.withdraw(self._id)
        except (RuntimeClosedError, BaseExceptionGroup) as exc:
            if not _swallow_closed(exc):
                raise
            logger.debug("withdraw dropped for {}: runtime closed", self._id)
            return False
        return True

    async def _reply_default(self, text: str | AsyncIterable[str]) -> str:
        turn = CURRENT_TURN.get()
        outputs = self._outputs()
        if outputs is None:
            return ""
        spoken = ""
        try:
            if turn is None:
                spoken = text if isinstance(text, str) else "".join([part async for part in text])
                await self._say_default(spoken)
                return spoken
            parts: list[str] = []
            async with outputs.reply(self._id, request_id=turn.id, timestamp_us=turn.timestamp_us) as reply:
                if isinstance(text, str):
                    parts.append(text)
                    await reply.send(text)
                else:
                    async for fragment in text:
                        parts.append(fragment)
                        await reply.send(fragment)
            spoken = "".join(parts)
        except (RuntimeClosedError, BaseExceptionGroup) as exc:
            if not _swallow_closed(exc):
                raise
            logger.debug("reply dropped for {}: runtime closed", self._id)
        return spoken

    # ------------------------------------------------------------------ tasks

    def spawn(
        self, coro: Coroutine[Any, Any, T], *, key: str | None = None, supersede: bool = False, detached: bool = False
    ) -> asyncio.Task[T]:
        """Run ``coro`` as this participant's background work, in its scope; traced as a task.

        ``key`` names the task: with ``supersede`` a live task under the same key is cancelled first, without it a live
        one raises ``ValueError``. ``detached``: the work is not part of the turn that started it (no current turn
        inside), so what it says is said on the assistant's own initiative rather than as part of that turn's answer.
        Raises ``RuntimeError`` once the scope stopped admitting tasks.
        """
        work = _tracing.in_task(_detach(coro) if detached else coro, key)
        return self._scope.spawn(work, key=key, supersede=supersede)

    async def cancel(self, key: str) -> bool:
        """Cancel and drain the task spawned under ``key``; return whether one was live."""
        return await self._scope.cancel(key)


async def _detach(coro: Coroutine[Any, Any, T]) -> T:
    CURRENT_TURN.set(None)  # this task's own context: the turn that started it is not its turn
    return await coro
