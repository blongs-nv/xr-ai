# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real ``Hub`` component over a stub ``ProcessorEndpoint`` (C-T3, G2): frames, connection events, no hub process.

``FakeHub`` is covered elsewhere; these tests run the real
component, so a wiring bug in it (not only in its fake) is visible.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

from xr_ai_next import Component
from xr_ai_next._base.app import App, AppConfig
from xr_ai_next._base.enrollment import Connection, PerConnection, Persistent
from xr_ai_next._base.hub import Hub
from xr_ai_next._base.participant import Participant
from xr_ai_next._base.scope import ScopeTimeouts
from xr_ai_next._base.testing import test_session


class FakeEndpoint:
    connected_participants: frozenset[str] = frozenset()  # what the capture source checks; nobody to ask here

    def __init__(self) -> None:
        self.on_frame_cb = None
        self.participant_cbs: list = []
        self.image_cbs: list = []
        self.requests: list = []
        self.stopped = False
        self.closed = False
        self.running = asyncio.Event()
        self.result = "frame"

    def on_frame(self, cb) -> None:
        self.on_frame_cb = cb

    def on_participant(self, cb) -> None:
        self.participant_cbs.append(cb)  # the Hub's own, and the picture source's

    async def on_participant_cb(self, event) -> None:
        for cb in self.participant_cbs:
            await cb(event)

    def _on_image_capture(self, cb):
        self.image_cbs.append(cb)
        return lambda: None

    async def run(self) -> None:
        self.running.set()
        await asyncio.Event().wait()

    async def request_frame(self, signal, timeout: float = 2.0):
        self.requests.append(signal)
        if self.result is None:
            return None
        return SimpleNamespace(seq=signal.seq, pts_us=signal.pts_us, width=4, height=2, data=b"raw")

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    # the hub side: what the real endpoint reports about participants
    async def join(self, pid: str = "p1", session: str = "s1") -> None:
        await self.on_participant_cb(SimpleNamespace(participant_id=pid, joined=True, participant_session_id=session))

    async def leave(self, pid: str = "p1", session: str = "s1") -> None:
        await self.on_participant_cb(SimpleNamespace(participant_id=pid, joined=False, participant_session_id=session))


def signal(pid: str = "p1", track: str = "t1", seq: int = 1, pts_us: int | None = None):
    pts = time.time_ns() // 1000 if pts_us is None else pts_us
    return SimpleNamespace(participant_id=pid, track_id=track, seq=seq, pts_us=pts, width=4, height=2)


def make(enrollment=None, **kwargs) -> tuple[App, FakeEndpoint]:  # noqa: ANN001
    ep = FakeEndpoint()
    timeouts = ScopeTimeouts(drain_s=0.5, hook_s=0.5, close_s=0.5)
    app = App(AppConfig(config_dir=Path("."), timeouts=timeouts), enrollment=enrollment or PerConnection(admit=None))
    kwargs.setdefault("min_interval_s", 0.05)
    hub = app.add(Hub(**kwargs))
    hub._endpoint_factory = lambda: ep
    hub._feed_options = {"encode": lambda d: (b"jpg" + d.data, "image/jpeg")}
    return app, ep


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0.01)


async def test_signal_after_join_reaches_p_frames() -> None:
    app, ep = make()
    await app.start()
    try:
        await ep.join()
        p = app._participants.get("p1")
        assert await p.frames._fresh() is None
        await ep.on_frame_cb(signal(seq=7))
        frame = await p.frames._next(timeout_s=1.0)
        assert frame is not None
        assert (frame.image, frame.mime_type, frame.width, frame.height) == (b"jpgraw", "image/jpeg", 4, 2)
        assert frame.sequence == 1
        assert abs(frame.timestamp_us - time.time_ns() // 1000) < 5_000_000
        await asyncio.sleep(0.06)
        await ep.on_frame_cb(signal(seq=8))
        assert (await p.frames._next(timeout_s=1.0)).sequence == 2
    finally:
        await app.stop()


async def test_signal_before_join_or_for_other_participant_is_ignored() -> None:
    app, ep = make()
    await app.start()
    try:
        await ep.on_frame_cb(signal("p1"))
        await ep.join()
        await ep.on_frame_cb(signal("other"))
        await settle()
        assert ep.requests == []
    finally:
        await app.stop()


async def _requests_until(ep: FakeEndpoint, count: int, timeout_s: float = 30.0) -> None:
    """Wait on the fetch count itself, not on a wall-clock guess of when the throttle fires."""
    async with asyncio.timeout(timeout_s):
        while len(ep.requests) < count:
            await asyncio.sleep(0.005)


async def test_throttle_fetches_pixels_once_per_interval_with_freshest_signal() -> None:
    interval = 1.0  # generous: the burst below must fit inside one interval even on a loaded machine
    app, ep = make(min_interval_s=interval)
    await app.start()
    try:
        await ep.join()
        await ep.on_frame_cb(signal(seq=1))
        await _requests_until(ep, 1)
        assert [r.seq for r in ep.requests] == [1]
        for seq in range(2, 30):  # a 30 signal burst inside one interval
            await ep.on_frame_cb(signal(seq=seq, pts_us=time.time_ns() // 1000 + seq))
            await asyncio.sleep(0)
        assert len(ep.requests) == 1
        await _requests_until(ep, 2)  # the throttle releases once the interval has passed
        assert [r.seq for r in ep.requests] == [1, 29]  # the freshest, not every signal
        await asyncio.sleep(interval * 1.5)
        assert len(ep.requests) == 2  # idle camera: no polling
    finally:
        await app.stop()


async def test_freshest_track_wins_and_configured_track_is_followed() -> None:
    app, ep = make()
    await app.start()
    try:
        await ep.join()
        now = time.time_ns() // 1000
        await ep.on_frame_cb(signal(track="old", pts_us=now - 1000))
        await ep.on_frame_cb(signal(track="new", pts_us=now))
        await settle()
        assert [r.track_id for r in ep.requests] == ["new"]
    finally:
        await app.stop()
    app, ep = make(track_id="screen")
    await app.start()
    try:
        await ep.join()
        await ep.on_frame_cb(signal(track="camera"))
        await settle()
        assert ep.requests == []
        await ep.on_frame_cb(signal(track="screen"))
        await settle()
        assert [r.track_id for r in ep.requests] == ["screen"]
    finally:
        await app.stop()


async def test_missing_pixels_and_encode_errors_do_not_end_the_feed() -> None:
    app, ep = make()
    await app.start()
    try:
        await ep.join()
        p = app._participants.get("p1")
        ep.result = None
        await ep.on_frame_cb(signal())
        await settle()
        assert await p.frames._fresh() is None
        ep.result = "frame"
        await asyncio.sleep(0.06)
        await ep.on_frame_cb(signal(seq=2))
        assert await p.frames._next(timeout_s=1.0) is not None
    finally:
        await app.stop()


async def test_clock_skew_pts_far_from_wall_clock_falls_back_to_receive_time() -> None:
    app, ep = make()
    await app.start()
    try:
        await ep.join()
        p = app._participants.get("p1")
        await ep.on_frame_cb(signal(pts_us=5_000_000))  # stream-relative pts
        frame = await p.frames._next(timeout_s=1.0)
        assert await p.frames._fresh(max_age_s=2.0) is frame
        assert abs(frame.timestamp_us - time.time_ns() // 1000) < 5_000_000
    finally:
        await app.stop()
    app, ep = make(clock="receive")
    await app.start()
    try:
        await ep.join()
        p = app._participants.get("p1")
        skewed = time.time_ns() // 1000 - 600 * 1_000_000  # device clock ten minutes behind
        await ep.on_frame_cb(signal(pts_us=skewed))
        frame = await p.frames._next(timeout_s=1.0)
        assert await p.frames._fresh(max_age_s=2.0) is frame
    finally:
        await app.stop()


async def test_leave_closes_the_stream_and_ends_the_pump() -> None:
    app, ep = make()
    await app.start()
    try:
        await ep.join()
        p = app._participants.get("p1")
        await ep.on_frame_cb(signal())
        assert await p.frames._next(timeout_s=1.0) is not None
        feed = app._components[0]._feed
        await ep.leave()  # PerConnection: the last disconnect un-enrolls
        assert app._participants.get("p1") is None
        assert await p.frames._fresh() is None
        assert await p.frames._next(timeout_s=0.5) is None
        assert feed._pumps == {}
        requests = len(ep.requests)
        await ep.on_frame_cb(signal(seq=9))
        await settle()
        assert len(ep.requests) == requests
    finally:
        await app.stop()


async def test_stop_closes_the_endpoint_and_ends_the_run_task() -> None:
    app, ep = make()
    await app.start()
    await ep.running.wait()
    await ep.join()
    await app.stop()
    assert ep.stopped and ep.closed
    assert not [t for t in app._app_tasks if not t.done()]
    assert not any(scope._tasks for scope in app._component_scopes.values())  # the run task was cancelled and drained


def test_encode_frame_makes_jpeg_from_raw_pixels() -> None:
    from xr_ai_hub import PixelFormat
    from xr_ai_next._base._frames_feed import encode_frame

    data = SimpleNamespace(width=4, height=2, fmt=PixelFormat.RGB24, data=bytes(4 * 2 * 3))
    image, mime = encode_frame(data)
    assert mime == "image/jpeg" and image[:2] == b"\xff\xd8"


async def test_a_failing_frame_signal_is_logged_and_does_not_exit_the_process(monkeypatch) -> None:
    """The real endpoint calls ``os._exit(1)`` when a callback raises; the feed's callback contains its errors."""
    import xr_ai_hub._processor as processor_module
    from loguru import logger
    from xr_ai_hub import ProcessorEndpoint, Subscribe
    from xr_ai_next._base._frames_feed import HubFrameFeed

    exits: list[int] = []
    monkeypatch.setattr(processor_module.os, "_exit", exits.append)
    logged: list[str] = []
    sink = logger.add(lambda m: logged.append(m.record["message"]), level="ERROR")
    endpoint = ProcessorEndpoint("inproc://feed-sub", "inproc://feed-push", filter=Subscribe.VIDEO)
    try:
        feed = HubFrameFeed(endpoint)
        endpoint._spawn(feed._on_signal(object()))  # the endpoint's own dispatch path; no participant_id: raises
        endpoint._spawn(feed._on_signal(signal()))  # an unknown participant: ignored
        await asyncio.sleep(0.01)

        async def raising() -> None:
            raise RuntimeError("unguarded")

        endpoint._spawn(raising())  # control: an unguarded callback does reach os._exit
        await asyncio.sleep(0.01)
    finally:
        logger.remove(sink)
        endpoint.close()
    assert exits == [1]
    assert len([m for m in logged if m.startswith("frame signal failed")]) == 1


# --- the Hub component itself: connection events, enrollment, frames across a disconnect (C-T3, G2) -----------------


class Watcher(Component):
    """An app with no Voice: presence still arrives (C-T3)."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[str] = []

    async def on_participant_enrolled(self, p: Participant) -> None:
        self.log.append(f"enrolled:{p.id}:{','.join(c.session_id for c in p._connections)}")

    async def on_participant_unenrolled(self, p: Participant) -> None:
        self.log.append(f"unenrolled:{p.id}")


async def test_presence_arrives_through_the_real_hub_with_no_voice_in_the_app() -> None:
    app, ep = make()
    watcher = app.add(Watcher())
    await app.start()
    try:
        await ep.join("alice", "sess-1")
        p = app._participants.get("alice")
        assert p is not None and p.connected and p.id == "alice"
        await ep.leave("alice", "sess-1")
        assert app._participants.get("alice") is None and not p.connected
        assert watcher.log == ["enrolled:alice:sess-1", "unenrolled:alice"]
    finally:
        await app.stop()


async def test_a_new_session_of_an_identity_supersedes_the_old_one_and_keeps_the_participant() -> None:
    app, ep = make(Persistent(keep_for_s=None, admit=None))
    watcher = app.add(Watcher())
    await app.start()
    try:
        await ep.join("alice", "s1")
        p = app._participants.get("alice")
        await ep.join("alice", "s1")  # the endpoint already de-duplicates; the component does too
        await ep.join("alice", "s2")  # the endpoint replaces s1 and will never report it leaving
        assert watcher.log == ["enrolled:alice:s1"]
        # s1 was disconnected before s2 connected, so the participant's connections never go stale
        assert app._participants.get("alice") is p and p.connected and p._connections == [Connection("alice", "s2")]
        await ep.leave("alice", "s2")
        assert app._participants.get("alice") is p and not p.connected  # Persistent keeps it
        await ep.leave("alice", "s2")  # a repeated leave is ignored
        assert app._participants.get("alice") is p and watcher.log == ["enrolled:alice:s1"]
    finally:
        await app.stop()


async def test_a_rejected_connection_never_becomes_a_participant_and_its_leave_is_ignored() -> None:
    app, ep = make(PerConnection(admit=lambda c: c.identity != "mallory"))
    await app.start()
    try:
        await ep.join("mallory")
        assert app._participants.get("mallory") is None
        await ep.on_frame_cb(signal("mallory"))
        await ep.leave("mallory")
        await settle()
        assert ep.requests == []
    finally:
        await app.stop()


async def test_frames_survive_a_disconnect_of_a_persistent_participant_and_resume_on_reconnect() -> None:
    app, ep = make(Persistent(keep_for_s=None, admit=None))
    await app.start()
    try:
        await ep.join()
        p = app._participants.get("p1")
        await ep.on_frame_cb(signal(seq=1))
        assert await p.frames._next(timeout_s=1.0) is not None
        await ep.leave()
        assert not p.connected and await p.frames._fresh() is None
        requests = len(ep.requests)
        await ep.on_frame_cb(signal(seq=2))
        await settle()
        assert len(ep.requests) == requests  # nobody is connected: no pixels fetched
        await ep.join(session="s2")
        await asyncio.sleep(0.06)
        await ep.on_frame_cb(signal(seq=3))
        frame = await p.frames._next(timeout_s=1.0)
        assert frame is not None and p.connected  # the same stream, open again
    finally:
        await app.stop()


async def test_a_hub_that_cannot_be_reached_is_built_in_on_start_not_at_construction() -> None:
    def boom():
        raise AssertionError("an endpoint was opened")

    hub = Hub()
    hub._endpoint_factory = boom
    app = App(AppConfig(config_dir=Path(".")), enrollment=PerConnection(admit=None))
    app.add(hub)  # nothing opened yet
    async with test_session(app):  # a session replaces the Hub, so nothing is ever opened
        pass


class DispatchingEndpoint(FakeEndpoint):
    """Dispatches every callback with ``create_task`` from inside ``run()``, as ``ProcessorEndpoint._spawn`` does."""

    def __init__(self) -> None:
        super().__init__()
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.dispatched: list[asyncio.Task] = []

    async def run(self) -> None:
        self.running.set()
        while True:
            callback, arg = await self.inbox.get()
            self.dispatched.append(asyncio.create_task(callback(arg)))


async def test_callbacks_dispatched_as_sibling_tasks_enroll_every_identity() -> None:
    """Two connection events arrive while the first enrollment is still inside a slow hook.

    The endpoint dispatches each callback as a sibling task of its run task; each runs as component work with its own
    Relay scope stack, and the component publishes them in the order received.
    """

    class Slow(Component):
        async def on_participant_enrolled(self, p: Participant) -> None:
            await asyncio.sleep(0.03)  # the CONNECTED publish stays open meanwhile

    ep = DispatchingEndpoint()
    app = App(AppConfig(config_dir=Path(".")), enrollment=PerConnection(admit=None))
    hub = app.add(Hub())
    hub._endpoint_factory = lambda: ep
    app.add(Slow())
    await app.start()
    try:
        await ep.running.wait()
        for pid in ("p1", "p2"):
            join = SimpleNamespace(participant_id=pid, joined=True, participant_session_id=f"s-{pid}")
            await ep.inbox.put((ep.on_participant_cb, join))
            await asyncio.sleep(0.01)  # the first enrollment is now inside the slow hook
        await asyncio.gather(*ep.dispatched, return_exceptions=False)
        assert [p.id for p in app.participants] == ["p1", "p2"]
        assert all(p.connected for p in app.participants)  # no connection event was lost
    finally:
        await app.stop()
