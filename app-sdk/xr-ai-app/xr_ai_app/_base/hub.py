# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``Hub`` component: the app's connection to the device hub.

It owns one ``ProcessorEndpoint`` at the standard local sockets and does two things:

- **Connection events.** ``ProcessorEndpoint.on_participant`` becomes the typed ``PARTICIPANT_CONNECTED`` and
  ``PARTICIPANT_DISCONNECTED`` events, with ``Connection(identity, session_id)``. This component is their only publisher
  (R8.4); what they mean for participants is the app's enrollment setting.
- **Frames.** Each enrolled participant gets a pump (a task in its scope) that fills ``p.frames`` while it is connected.

The endpoint is built in ``on_start``, never at construction, so a sample may build ``Hub()`` at import and a test
session (which replaces it with ``FakeHub``) never opens a socket.

The endpoint tracks one active session per identity: a join with a new session id replaces the old one, and the old
session's leave is never reported. The component therefore publishes ``PARTICIPANT_DISCONNECTED`` for a superseded
session right before ``PARTICIPANT_CONNECTED`` for the new one, so the participant's connections never go stale. Several
simultaneous connections of one identity are exercised by ``FakeHub`` only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from loguru import logger
from xr_ai_runtime import RuntimeClosedError

from ._frames_feed import HubFrameFeed, hub_endpoint
from .component import Component
from .enrollment import Connection
from .participant import Participant
from .topics import (
    PARTICIPANT_CONNECTED,
    PARTICIPANT_DISCONNECTED,
    ParticipantConnected,
    ParticipantDisconnected,
)

__all__ = ["Hub"]


class Hub(Component):
    """Connection to the device hub: frames into ``p.frames``, and presence.

    ``min_interval_s``, ``track_id``, ``fetch_timeout_s`` and ``clock`` tune the frame pump exactly as they tuned the
    implicit feed they replace: pixels are fetched at most once per ``min_interval_s`` (the freshest signal at that
    moment); ``track_id=None`` follows the track with the newest ``pts_us``; ``clock="pts"`` trusts the device clock
    when it is plausible and ``"receive"`` always stamps the local receive time.
    """

    def __init__(
        self,
        *,
        min_interval_s: float = 0.5,
        track_id: str | None = None,
        fetch_timeout_s: float = 2.0,
        clock: Literal["pts", "receive"] = "pts",
    ) -> None:
        self._tuning: dict[str, Any] = {
            "min_interval_s": min_interval_s,
            "track_id": track_id,
            "fetch_timeout_s": fetch_timeout_s,
            "clock": clock,
        }
        self._endpoint_factory: Callable[[], Any] = hub_endpoint
        """Private seam (tests): ``factory() -> endpoint``; the default opens the standard local sockets."""
        self._feed_options: dict[str, Any] = {}
        """Private seam (tests): extra ``HubFrameFeed`` arguments, such as ``encode``."""
        self._feed: HubFrameFeed | None = None
        self._active: dict[str, Connection] = {}
        self._order = asyncio.Lock()

    async def on_start(self) -> None:
        """Build the endpoint, register its callbacks and run it for the app's lifetime."""
        endpoint = self._endpoint_factory()
        self._feed = HubFrameFeed(endpoint, **self._tuning, **self._feed_options)
        endpoint.on_participant(self._isolated(self._on_participant))
        self.spawn(self._run(), key="hub")

    def _isolated(self, handler: Callable[[Any], Awaitable[None]]) -> Callable[[Any], Awaitable[None]]:
        """Run each endpoint callback as component work, so it has its own Relay scope stack.

        The endpoint dispatches every callback as a sibling task of its run task. Siblings that publish concurrently
        (connection events held open across the enrollment hooks) would otherwise share one scope stack and pop it out
        of order. ``Component.spawn`` forks the stack; awaiting the task keeps the order.
        """

        async def run(message: Any) -> None:
            try:
                task = self.spawn(handler(message))
            except RuntimeError:  # the app is stopping
                logger.debug("hub callback dropped: the app is stopping")
                return
            await task

        return run

    async def _run(self) -> None:
        # Connection events that arrive before the app admits participants would be dropped: start receiving after.
        await self.app._admitted.wait()
        assert self._feed is not None  # noqa: S101 - set by on_start, which spawned this task
        await self._feed.run()

    async def on_participant_enrolled(self, p: Participant) -> None:
        """Start the frame pump of ``p`` in its scope (it ends with the scope at un-enrollment)."""
        if self._feed is not None:
            self._feed.attach(p)

    async def on_stop(self) -> None:
        """Stop and close the endpoint (the run task was cancelled before the stop hooks)."""
        feed, self._feed = self._feed, None
        if feed is not None:
            feed.close()

    async def _on_participant(self, event: Any) -> None:
        """A ``ParticipantEvent`` from the endpoint becomes a connection event (published in the order received)."""
        connection = Connection(event.participant_id, getattr(event, "participant_session_id", "") or "")
        async with self._order:
            if event.joined:
                old = self._active.get(connection.identity)
                if old == connection:
                    return
                if old is not None:  # the endpoint replaced the session and will never report the old one leaving
                    await self._publish(PARTICIPANT_DISCONNECTED, ParticipantDisconnected(connection=old), old)
                self._active[connection.identity] = connection
                await self._publish(PARTICIPANT_CONNECTED, ParticipantConnected(connection=connection), connection)
                return
            active = self._active.pop(connection.identity, None)
            if active is not None:
                await self._publish(PARTICIPANT_DISCONNECTED, ParticipantDisconnected(connection=active), active)

    async def _publish(self, topic: Any, message: Any, connection: Connection) -> None:
        try:
            await self.app._runtime.publish(topic, message, participant_id=connection.identity, source="hub")
        except RuntimeClosedError:
            logger.debug("{} for {} dropped: runtime closed", topic.name, connection.identity)
