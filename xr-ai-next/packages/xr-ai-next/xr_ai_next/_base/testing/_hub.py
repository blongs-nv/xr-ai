# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``FakeHub``: a ``Hub`` replacement, a component that connects, disconnects and reconnects identities, feeds frames.

Added to an app by ``test_session`` (or ``test_session(app, replace={Hub: FakeHub()})``), it publishes the same typed
connection events as the real ``Hub`` component (``PARTICIPANT_CONNECTED/DISCONNECTED``, R8.4), pushes frames into the
current participant's ``p.frames``, holds the picture a client in on-demand camera mode would take, stands in typed
text for speech and records what the app says back, and the agent status (``processing``, ``idle``) it reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

from xr_ai_interaction import UserQuery
from xr_ai_runtime import RuntimeClosedError

from .. import _virtual
from ..component import Component
from ..enrollment import Connection
from ..frames import Frame
from ..topics import (
    PARTICIPANT_CONNECTED,
    PARTICIPANT_DISCONNECTED,
    USER_QUERY,
    ParticipantConnected,
    ParticipantDisconnected,
)

if TYPE_CHECKING:
    from ..app import App
    from ..participant import Participant


class HubListener(Protocol):
    """What can be attached to a :class:`FakeHub` to receive hub events.

    The hub, added as a component, attaches an adapter that turns these into runtime events and participant
    updates; a test (or a reference app behind ``_install_test_doubles``) may attach any object with these four
    coroutines. A listener that also has ``on_hub_connected(connection)`` / ``on_hub_disconnected(connection)``
    is told about every connection; one without them is told ``on_hub_join`` for the first connection of an
    identity and ``on_hub_leave`` when its last one ends.
    """

    async def on_hub_join(self, participant_id: str) -> None:
        """A participant joined."""

    async def on_hub_leave(self, participant_id: str) -> None:
        """A participant left."""

    async def on_hub_frame(self, participant_id: str, frame: Frame) -> None:
        """A participant's camera delivered ``frame``."""

    async def on_hub_query(self, participant_id: str, text: str, timestamp_us: int) -> None:
        """A participant sent a text query in place of speech."""


@dataclass(frozen=True, slots=True)
class HubEvent:
    """One thing that crossed the fake hub, in order, for assertions."""

    kind: str
    """``join`` (first connection of an identity), ``connect`` (another one), ``leave`` (last disconnect),
    ``disconnect`` (another), ``frame``, ``picture`` (held, not streamed), ``capture`` (the app asked for a picture;
    ``detail`` is the frame handed back, or ``None``), ``query``, ``output``, ``status`` or ``agent_status``."""

    participant_id: str
    detail: Any = None


@dataclass(slots=True)
class _Output:
    text: str
    urgent: bool = False


@dataclass(slots=True)
class _Participant:
    next_sequence: int = 0
    picture: Frame | None = None
    spoken: list[_Output] = field(default_factory=list)
    statuses: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    agent_status: list[str] = field(default_factory=list)


class _RuntimeBridge:
    """The listener a ``FakeHub`` attaches when it is added to an app: device events become runtime events."""

    def __init__(self, app: App) -> None:
        self._app = app

    async def on_hub_join(self, participant_id: str) -> None:  # pragma: no cover - on_hub_connected is used instead
        """Unused: the bridge is told every connection."""

    async def on_hub_leave(self, participant_id: str) -> None:  # pragma: no cover
        """Unused: the bridge is told every disconnection."""

    async def on_hub_connected(self, connection: Connection) -> None:
        await self._app._runtime.publish(
            PARTICIPANT_CONNECTED,
            ParticipantConnected(connection=connection),
            participant_id=connection.identity,
            source="fake-hub",
        )

    async def on_hub_disconnected(self, connection: Connection) -> None:
        await self._app._runtime.publish(
            PARTICIPANT_DISCONNECTED,
            ParticipantDisconnected(connection=connection),
            participant_id=connection.identity,
            source="fake-hub",
        )

    async def on_hub_frame(self, participant_id: str, frame: Frame) -> None:
        p = self._app._participants.get(participant_id)
        if p is not None and p._scope.admitting and p.connected:
            p.frames._push(frame)

    async def on_hub_query(self, participant_id: str, text: str, timestamp_us: int) -> None:
        await self._app._runtime.publish(
            USER_QUERY,
            UserQuery(text=text, timestamp_us=timestamp_us),
            participant_id=participant_id,
            source="fake-hub",
        )


class FakeHub(Component):
    """In-memory hub, a ``Hub`` replacement: identities connect, disconnect and reconnect, frames are pushed per
    participant, and text queries stand in for speech. A participant's camera may instead hold a picture
    (:meth:`hold_picture`) that it hands over only when the app asks for one, as a client in on-demand camera mode does.
    App output (speech, status and agent status) is recorded per participant.

    An identity may have several connections (devices), each with its own session id; it counts as connected while
    it has one. Like the real runtime, publishing after :meth:`close` raises ``RuntimeClosedError``, so a task that
    outlives ``App.stop()`` is visible to a test.
    """

    def __init__(self) -> None:
        self._listeners: list[HubListener] = []
        self._bridge: _RuntimeBridge | None = None
        self._sessions: dict[str, list[str]] = {}
        self._session_counter = 0
        self._participants: dict[str, _Participant] = {}
        self._history: dict[str, _Participant] = {}
        self._events: list[HubEvent] = []
        self._closed = False

    # -- component side -------------------------------------------------------------------

    async def on_start(self) -> None:
        """Attach to the app's runtime: from now on device events are published as the real ``Hub`` would."""
        self._bridge = _RuntimeBridge(self.app)
        self.attach(self._bridge)

    async def on_participant_enrolled(self, p: Participant) -> None:
        """Let ``p.frames`` ask this hub for a picture (answered from :meth:`hold_picture`)."""
        p.frames._capture = lambda: self._take_picture(p.id)

    async def on_stop(self) -> None:
        """Stop delivering device events."""
        if self._bridge is not None and self._bridge in self._listeners:
            self.detach(self._bridge)

    # -- app side --------------------------------------------------------------------------

    def attach(self, listener: HubListener) -> None:
        """Deliver hub events to ``listener``; listeners are called in attach order."""
        self._listeners.append(listener)

    def detach(self, listener: HubListener) -> None:
        """Stop delivering events to ``listener``."""
        self._listeners.remove(listener)

    def close(self) -> None:
        """Close the hub; later :meth:`publish_output` and :meth:`publish_status` raise ``RuntimeClosedError``."""
        self._closed = True

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` was called."""
        return self._closed

    def publish_output(self, participant_id: str, text: str, *, urgent: bool = False) -> None:
        """Record speech the app produced for ``participant_id``."""
        self._check_open()
        self._record(participant_id).spoken.append(_Output(text, urgent))
        self._events.append(HubEvent("output", participant_id, text))

    def publish_status(self, participant_id: str, text: str, **detail: Any) -> None:
        """Record a status the app reported for ``participant_id``."""
        self._check_open()
        self._record(participant_id).statuses.append((text, dict(detail)))
        self._events.append(HubEvent("status", participant_id, (text, dict(detail))))

    def publish_agent_status(self, participant_id: str, status: str) -> None:
        """Record the agent status the app reported for ``participant_id`` (``processing`` or ``idle``)."""
        self._check_open()
        self._record(participant_id).agent_status.append(status)
        self._events.append(HubEvent("agent_status", participant_id, status))

    # -- device side -----------------------------------------------------------------------

    async def connect(self, identity: str = "p1", *, session_id: str | None = None) -> Connection:
        """A device connects as ``identity``; a second device with the same identity is another connection."""
        if session_id is None:
            self._session_counter += 1
            session_id = f"{identity}-s{self._session_counter}"
        sessions = self._sessions.setdefault(identity, [])
        if session_id in sessions:
            raise ValueError(f"session {session_id!r} of {identity!r} is already connected")
        first = not sessions
        sessions.append(session_id)
        if first:
            self._participants[identity] = _Participant()
            self._history.setdefault(identity, self._participants[identity])
        connection = Connection(identity, session_id)
        self._events.append(HubEvent("join" if first else "connect", identity, connection))
        for listener in list(self._listeners):
            handler = getattr(listener, "on_hub_connected", None)
            if handler is not None:
                await handler(connection)
            elif first:
                await listener.on_hub_join(identity)
        return connection

    async def disconnect(self, identity: str = "p1", session_id: str | None = None) -> None:
        """A device disconnects: ``session_id``, or the most recently connected one when ``None``."""
        sessions = self._sessions.get(identity)
        if not sessions:
            raise KeyError(f"participant {identity!r} is not connected")
        if session_id is None:
            session_id = sessions[-1]
        if session_id not in sessions:
            raise KeyError(f"session {session_id!r} of {identity!r} is not connected")
        sessions.remove(session_id)
        last = not sessions
        if last:
            del self._sessions[identity]
            del self._participants[identity]
        connection = Connection(identity, session_id)
        self._events.append(HubEvent("leave" if last else "disconnect", identity, connection))
        for listener in list(self._listeners):
            handler = getattr(listener, "on_hub_disconnected", None)
            if handler is not None:
                await handler(connection)
            elif last:
                await listener.on_hub_leave(identity)

    async def reconnect(self, identity: str = "p1", *, session_id: str | None = None) -> Connection:
        """Every connection of ``identity`` ends (when it has any) and a new session connects."""
        while identity in self._sessions:
            await self.disconnect(identity)
        return await self.connect(identity, session_id=session_id)

    async def join(self, participant_id: str = "p1") -> None:
        """Announce a participant: its first connection."""
        if participant_id in self._participants:
            raise ValueError(f"participant {participant_id!r} already joined")
        await self.connect(participant_id)

    async def leave(self, participant_id: str = "p1") -> None:
        """Announce that a participant left: every connection of the identity ends."""
        if participant_id not in self._participants:
            raise KeyError(f"participant {participant_id!r} is not connected")
        while participant_id in self._sessions:
            await self.disconnect(participant_id)

    async def push_frame(self, frame: Frame, participant_id: str = "p1") -> None:
        """Deliver one camera frame."""
        self._require(participant_id)
        self._events.append(HubEvent("frame", participant_id, frame))
        for listener in list(self._listeners):
            await listener.on_hub_frame(participant_id, frame)

    def hold_picture(self, frame: Frame | None, participant_id: str = "p1") -> None:
        """The camera of ``participant_id`` holds ``frame`` without streaming it; the app gets it only by asking
        (``p.frames.current()`` with no fresh frame). ``None``: the camera cannot take a picture (the default)."""
        self._require(participant_id).picture = frame
        self._events.append(HubEvent("picture", participant_id, frame))

    async def _take_picture(self, participant_id: str) -> Frame | None:
        state = self._participants.get(participant_id)
        frame = None
        if state is not None and state.picture is not None:
            frame = replace(state.picture, timestamp_us=_virtual.wall_us())
        self._events.append(HubEvent("capture", participant_id, frame))
        return frame

    async def text_query(self, text: str, participant_id: str = "p1") -> None:
        """Deliver a typed query as if it were accepted speech."""
        self._require(participant_id)
        self._events.append(HubEvent("query", participant_id, text))
        timestamp_us = _virtual.wall_us()
        for listener in list(self._listeners):
            await listener.on_hub_query(participant_id, text, timestamp_us)

    def make_frame(
        self,
        participant_id: str = "p1",
        *,
        image: bytes = b"fake-jpeg",
        width: int = 64,
        height: int = 48,
        mime_type: str = "image/jpeg",
    ) -> Frame:
        """Return the next frame for ``participant_id`` with a monotonic ``sequence`` and timestamp."""
        state = self._require(participant_id)
        state.next_sequence += 1
        return Frame(
            image=image,
            timestamp_us=_virtual.wall_us(),
            width=width,
            height=height,
            mime_type=mime_type,
            sequence=state.next_sequence,
        )

    # -- inspection ------------------------------------------------------------------------

    @property
    def participants(self) -> frozenset[str]:
        """Ids of the participants currently connected."""
        return frozenset(self._participants)

    def connections(self, identity: str = "p1") -> list[Connection]:
        """The live connections of ``identity``, oldest first."""
        return [Connection(identity, sid) for sid in self._sessions.get(identity, [])]

    @property
    def events(self) -> list[HubEvent]:
        """Every join, leave, frame, query, output and status in the order it happened."""
        return list(self._events)

    def spoken(self, participant_id: str = "p1") -> list[str]:
        """Everything the app said to ``participant_id``, including after the participant left."""
        state = self._history.get(participant_id)
        return [o.text for o in state.spoken] if state else []

    def urgent(self, participant_id: str = "p1") -> list[bool]:
        """The ``urgent`` flag of each spoken message, aligned with :meth:`spoken`."""
        state = self._history.get(participant_id)
        return [o.urgent for o in state.spoken] if state else []

    def statuses(self, participant_id: str = "p1") -> list[tuple[str, dict[str, Any]]]:
        """Every ``(text, detail)`` status reported for ``participant_id``."""
        state = self._history.get(participant_id)
        return list(state.statuses) if state else []

    def agent_status(self, participant_id: str = "p1") -> list[str]:
        """Every agent status reported for ``participant_id``, in order (``processing`` while a turn is answered,
        ``idle`` after)."""
        state = self._history.get(participant_id)
        return list(state.agent_status) if state else []

    # -- internals -------------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeClosedError("FakeHub is closed")

    def _record(self, participant_id: str) -> _Participant:
        return self._history.setdefault(participant_id, _Participant())

    def _require(self, participant_id: str) -> _Participant:
        try:
            return self._participants[participant_id]
        except KeyError:
            raise KeyError(f"participant {participant_id!r} is not connected") from None
