# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FakeHub: joins, leaves, frames, text queries, recorded output and the closed-runtime behavior."""

from __future__ import annotations

import pytest
from xr_ai_app._base.testing import FakeHub
from xr_ai_runtime import RuntimeClosedError


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def on_hub_join(self, participant_id: str) -> None:
        self.events.append(("join", participant_id))

    async def on_hub_leave(self, participant_id: str) -> None:
        self.events.append(("leave", participant_id))

    async def on_hub_frame(self, participant_id: str, frame) -> None:
        self.events.append(("frame", participant_id, frame.sequence))

    async def on_hub_query(self, participant_id: str, text: str, timestamp_us: int) -> None:
        self.events.append(("query", participant_id, text, timestamp_us > 0))


async def test_events_reach_listeners_in_order_per_participant() -> None:
    hub, rec = FakeHub(), Recorder()
    hub.attach(rec)
    await hub.join("a")
    await hub.join("b")
    await hub.push_frame(hub.make_frame("a"), "a")
    await hub.push_frame(hub.make_frame("a"), "a")
    await hub.text_query("hello", "b")
    await hub.leave("a")
    assert rec.events == [
        ("join", "a"),
        ("join", "b"),
        ("frame", "a", 1),
        ("frame", "a", 2),
        ("query", "b", "hello", True),
        ("leave", "a"),
    ]
    assert hub.participants == {"b"}
    assert [e.kind for e in hub.events] == ["join", "join", "frame", "frame", "query", "leave"]


async def test_frame_sequences_are_per_participant_and_monotonic() -> None:
    hub = FakeHub()
    await hub.join("a")
    await hub.join("b")
    a1, a2, b1 = hub.make_frame("a"), hub.make_frame("a"), hub.make_frame("b")
    assert (a1.sequence, a2.sequence, b1.sequence) == (1, 2, 1)
    assert a2.timestamp_us >= a1.timestamp_us


async def test_unknown_or_duplicate_participants_are_rejected() -> None:
    hub = FakeHub()
    with pytest.raises(KeyError):
        await hub.text_query("hi", "ghost")
    with pytest.raises(KeyError):
        await hub.leave("ghost")
    await hub.join("a")
    with pytest.raises(ValueError, match="already joined"):
        await hub.join("a")


async def test_output_is_recorded_per_participant_and_survives_leave() -> None:
    hub = FakeHub()
    await hub.join("a")
    hub.publish_output("a", "one")
    hub.publish_output("a", "two", urgent=True)
    hub.publish_status("a", "busy", step=3)
    await hub.leave("a")
    assert hub.spoken("a") == ["one", "two"]
    assert hub.urgent("a") == [False, True]
    assert hub.statuses("a") == [("busy", {"step": 3})]
    assert hub.spoken("nobody") == []


async def test_publishing_after_close_raises_like_the_runtime() -> None:
    hub = FakeHub()
    await hub.join("a")
    hub.close()
    assert hub.closed
    with pytest.raises(RuntimeClosedError):
        hub.publish_output("a", "late")
    with pytest.raises(RuntimeClosedError):
        hub.publish_status("a", "late")


async def test_detached_listener_stops_receiving() -> None:
    hub, rec = FakeHub(), Recorder()
    hub.attach(rec)
    hub.detach(rec)
    await hub.join("a")
    assert rec.events == []
