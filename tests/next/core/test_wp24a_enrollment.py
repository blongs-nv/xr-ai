# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection events, enrollment policies, connections of a participant, and the test kit that drives them."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from xr_ai_next import Component
from xr_ai_next._base.app import App, AppConfig
from xr_ai_next._base.enrollment import Connection, PerConnection, Persistent
from xr_ai_next._base.participant import Participant
from xr_ai_next._base.scope import ScopeTimeouts
from xr_ai_next._base.testing import FakeHub, FakeModels, test_session
from xr_ai_next._base.topics import PARTICIPANT_CONNECTED, ParticipantConnected

TIMEOUTS = ScopeTimeouts(drain_s=0.2, hook_s=0.2, close_s=0.2)


def make_app(enrollment=None) -> App:  # noqa: ANN001
    return App(AppConfig(config_dir=Path("."), timeouts=TIMEOUTS), enrollment=enrollment)


class Recorder(Component):
    """Records its enrollment hooks."""

    def __init__(self, name: str = "A", log: list[str] | None = None) -> None:
        super().__init__()
        self.name = name
        self.log = log if log is not None else []

    async def on_participant_enrolled(self, p: Participant) -> None:
        self.log.append(f"{self.name}:enrolled:{p.id}:{p.connected}")

    async def on_participant_unenrolled(self, p: Participant) -> None:
        self.log.append(f"{self.name}:unenrolled:{p.id}")


async def until(condition, timeout_s: float = 5.0) -> None:  # noqa: ANN001
    """Wait on the condition itself, not on a guess of when it becomes true."""
    async with asyncio.timeout(timeout_s):
        while not condition():
            await asyncio.sleep(0.005)


# --- PerConnection -------------------------------------------------------------------------------------------------


async def test_per_connection_enrolls_on_connect_and_unenrolls_on_the_last_disconnect() -> None:
    app, rec = make_app(PerConnection(admit=None)), Recorder()
    app.add(rec)
    async with test_session(app) as s:
        await s.hub.connect("alice", session_id="a1")
        p = app._participants.get("alice")
        assert p is not None and p.connected and p.id == "alice"
        assert p._connections == [Connection("alice", "a1")]  # presence comes from the hub; the id is the identity
        assert rec.log == ["A:enrolled:alice:True"]
        await s.hub.disconnect("alice")
        assert app._participants.get("alice") is None and not p.connected
        assert rec.log[1:] == ["A:unenrolled:alice"]


async def test_a_second_connection_of_the_same_identity_is_the_same_participant() -> None:
    app, rec = make_app(PerConnection(admit=None)), Recorder()
    app.add(rec)
    async with test_session(app) as s:
        c1 = await s.hub.connect("alice")
        p = app._participants.get("alice")
        c2 = await s.hub.connect("alice")
        assert c1 != c2 and app._participants.get("alice") is p
        enrolled = [e for e in rec.log if "enrolled" in e and "un" not in e]
        assert enrolled == ["A:enrolled:alice:True"]
        await s.hub.disconnect("alice", c1.session_id)
        assert p.connected and app._participants.get("alice") is p  # still one device
        await s.hub.disconnect("alice", c2.session_id)
        assert app._participants.get("alice") is None


async def test_admit_rejects_a_connection_and_its_disconnect_is_ignored() -> None:
    seen: list[Connection] = []

    def admit(c: Connection) -> bool:
        seen.append(c)
        return c.identity != "mallory"

    app, rec = make_app(PerConnection(admit=admit)), Recorder()
    app.add(rec)
    async with test_session(app) as s:
        await s.hub.connect("mallory", session_id="m1")
        assert app._participants.get("mallory") is None
        await s.hub.disconnect("mallory")  # nothing to un-enroll, nothing raised
        await s.hub.connect("alice")
    assert [c.identity for c in seen] == ["mallory", "alice"]
    assert not any("enrolled:mallory" in e for e in rec.log)
    assert any(e.kind == "leave" and e.participant_id == "mallory" for e in s.hub.events)  # sent; the app ignored it


async def test_reenrolling_after_unenrollment_gives_a_fresh_participant() -> None:
    app = make_app(PerConnection(admit=None))
    async with test_session(app) as s:
        await s.hub.connect("alice")
        first = app._participants.get("alice")
        first.state.note = 1
        await s.hub.reconnect("alice")
        second = app._participants.get("alice")
        assert second is not first and "note" not in second.state and second.connected


# --- Persistent ----------------------------------------------------------------------------------------------------


async def test_persistent_keeps_the_participant_across_a_disconnect_and_a_reconnect_cancels_the_timer() -> None:
    app, rec = make_app(Persistent(keep_for_s=30.0, admit=None)), Recorder()
    app.add(rec)
    async with test_session(app) as s:
        await s.hub.connect("alice")
        p = app._participants.get("alice")
        p.state.visits = 1
        kept = p.spawn(asyncio.sleep(3600), key="work")  # p.spawn work survives a disconnect
        await s.hub.disconnect("alice")
        assert app._participants.get("alice") is p and not p.connected and not kept.done()
        assert not any("unenrolled" in e for e in rec.log)
        retain = [t for t in app._app_tasks if t.get_name() == "app:retain:alice"]
        assert len(retain) == 1  # the timer is an app task (R1)
        await s.hub.connect("alice")
        assert app._participants.get("alice") is p and p.connected and p.state.visits == 1
        assert retain[0].cancelled() or retain[0].cancelling() or retain[0].done()
        assert not kept.done()
        await s.hub.disconnect("alice")
    assert kept.cancelled()  # un-enrollment (here by app stop) cancels it


async def test_persistent_unenrolls_after_keep_for_s_and_cancels_participant_work() -> None:
    app, rec = make_app(Persistent(keep_for_s=0.05, admit=None)), Recorder()
    app.add(rec)
    async with test_session(app) as s:
        await s.hub.connect("alice")
        p = app._participants.get("alice")
        kept = p.spawn(asyncio.sleep(3600), key="work")
        await s.hub.disconnect("alice")
        await until(lambda: app._participants.get("alice") is None)
        assert kept.cancelled()
        assert rec.log[-1:] == ["A:unenrolled:alice"]


async def test_persistent_without_a_limit_keeps_the_participant_until_the_app_stops() -> None:
    app, rec = make_app(Persistent(keep_for_s=None, admit=None)), Recorder()
    app.add(rec)
    async with test_session(app) as s:
        await s.hub.connect("alice")
        await s.hub.disconnect("alice")
        await asyncio.sleep(0.05)
        assert app._participants.get("alice") is not None
        assert not [t for t in app._app_tasks if t.get_name().startswith("app:retain")]
    assert "A:unenrolled:alice" in rec.log  # the stop sequence un-enrolls everyone


async def test_frames_follow_the_current_connection_and_are_none_while_disconnected() -> None:
    app = make_app(Persistent(keep_for_s=None, admit=None))
    async with test_session(app) as s:
        await s.hub.connect("alice")
        p = app._participants.get("alice")
        await s.hub.push_frame(s.hub.make_frame("alice"), "alice")
        assert await p.frames._fresh() is not None
        await s.hub.disconnect("alice")
        assert await p.frames._fresh() is None  # nothing to show while nobody is connected...
        await s.hub.connect("alice")
        assert await p.frames._fresh() is None
        await s.hub.push_frame(s.hub.make_frame("alice"), "alice")
        assert await p.frames._fresh() is not None  # ...and the stream is still open for the next connection


async def test_a_second_device_keeps_the_participant_connected_when_the_first_leaves() -> None:
    app = make_app(Persistent(keep_for_s=None, admit=None))
    async with test_session(app) as s:
        c1 = await s.hub.connect("alice")
        await s.hub.connect("alice")
        p = app._participants.get("alice")
        await s.hub.push_frame(s.hub.make_frame("alice"), "alice")
        await s.hub.disconnect("alice", c1.session_id)
        assert p.connected and await p.frames._fresh() is not None


# --- component-lifetime work survives un-enrollment ------------------------------------------------------------------


async def test_component_spawn_survives_unenrollment_and_is_cancelled_at_stop() -> None:
    class Worker(Component):
        task: asyncio.Task[None] | None = None

        async def on_participant_enrolled(self, p: Participant) -> None:
            if self.task is None:
                self.task = self.spawn(asyncio.sleep(3600), key="bg")

    app, worker = make_app(PerConnection(admit=None)), Worker()
    app.add(worker)
    async with test_session(app) as s:
        await s.hub.connect("alice")
        await s.hub.disconnect("alice")
        assert app._participants.get("alice") is None
        assert worker.task is not None and not worker.task.done()
    assert worker.task.cancelled()


# --- no policy, and the test-kit conveniences ----------------------------------------------------------------------


async def test_a_hand_built_app_without_enrollment_ignores_connections_until_a_session_gives_it_one(caplog) -> None:  # noqa: ANN001
    app = make_app()
    assert app._enrollment is None
    await app.start()
    await app._runtime.publish(
        PARTICIPANT_CONNECTED, ParticipantConnected(connection=Connection("alice", "s")), participant_id="alice"
    )
    assert app._participants.get("alice") is None
    await app.stop()
    app2 = make_app()
    async with test_session(app2) as s:  # test-kit convenience, not an SDK default
        await s.hub.connect("alice")
        assert app2._participants.get("alice") is not None
        assert isinstance(app2._enrollment, PerConnection)


async def test_test_session_swaps_a_registered_hub_for_a_fake_hub() -> None:
    from xr_ai_next._base.hub import Hub

    app = make_app(PerConnection(admit=None))
    real = app.add(Hub())  # building it opens nothing: the endpoint is made in on_start
    fake = FakeHub()
    async with test_session(app, replace={Hub: fake}) as s:
        assert s.hub is fake and real not in app._components and fake in app._components
        await s.hub.connect("alice")
        assert app._participants.get("alice") is not None


async def test_test_session_adds_a_fake_hub_after_the_apps_own_components_when_it_has_none() -> None:
    app, rec = make_app(), Recorder()
    app.add(rec)
    async with test_session(app, models=FakeModels()) as s:
        assert app._components[0] is rec and app._components[-1] is s.hub


async def test_replace_of_an_unregistered_class_is_an_error() -> None:
    class Other(Component):
        pass

    app = make_app()
    with pytest.raises(ValueError, match="has no Other"):
        async with test_session(app, replace={Other: Other()}):
            pass


async def test_replace_swaps_any_added_component_by_class() -> None:
    class Real(Recorder):
        pass

    class Stand(Recorder):
        pass

    app = make_app()
    app.add(Real("real"))
    stand = Stand("stand")
    async with test_session(app, replace={Real: stand}) as s:
        await s.join("p1")
        assert stand.log == ["stand:enrolled:p1:True"]
    assert app._components[0] is stand


async def test_fake_hub_reconnect_ends_every_session_and_connects_a_new_one() -> None:
    hub = FakeHub()
    c1 = await hub.connect("alice")
    await hub.connect("alice")
    c3 = await hub.reconnect("alice")
    assert hub.connections("alice") == [c3] and c3 != c1
    assert [e.kind for e in hub.events] == ["join", "connect", "disconnect", "leave", "join"]
    with pytest.raises(KeyError):
        await hub.disconnect("alice", "nope")
    with pytest.raises(ValueError, match="already connected"):
        await hub.connect("alice", session_id=c3.session_id)


def test_policies_implement_the_enrollment_protocol_and_validate_keep_for_s() -> None:
    from xr_ai_next._base.enrollment import Enrollment

    assert isinstance(PerConnection(admit=None), Enrollment) and PerConnection(admit=None).keep_for_s == 0.0
    policy = Persistent(keep_for_s=2.5, admit=None)
    assert isinstance(policy, Enrollment) and policy.keep_for_s == 2.5 and policy.admit is None
    assert Persistent(keep_for_s=None, admit=None).keep_for_s is None
    with pytest.raises(ValueError, match="keep_for_s"):
        Persistent(keep_for_s=-1.0, admit=None)
    with pytest.raises(TypeError):
        PerConnection()  # type: ignore[call-arg]  # admit is required: no default policy, no default admission


async def test_a_task_spawned_in_the_enrollment_hook_may_publish_while_the_hook_is_still_running() -> None:
    """The spawned task has its own Relay scope stack, so its publishes cannot corrupt the enrolling delivery's."""

    class Chatty(Component):
        async def on_participant_enrolled(self, p: Participant) -> None:
            async def chatter() -> None:
                while True:
                    await p.say("tick")
                    await asyncio.sleep(0)

            p.spawn(chatter(), key="chatter")

    app = make_app()
    app.add(Chatty())
    app.add(Component())  # a later hook yields while the chatter runs
    async with test_session(app) as s:
        await s.join()
        await s.settle()
