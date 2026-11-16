# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501
"""The participants round on the live app-core names (``xr_ai_app._base``): component spawn and enrollment hooks,
``Connection``, the enrollment policies, the connection topics, and the removed presence forms."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect

import pytest
import xr_ai_app
import xr_ai_interaction
from pydantic import BaseModel, ValidationError
from xr_ai_app import Component
from xr_ai_app._base import enrollment, topics
from xr_ai_app._base.app import App
from xr_ai_app._base.participant import Participant


def _shape(fn: object) -> list[tuple[str, object, object]]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]  # type: ignore[arg-type]


AGENT_PARTICIPANT_SIGNATURES = {
    "Component.on_participant_enrolled": "(self, p: 'Participant') -> 'None'",
    "Component.on_participant_unenrolled": "(self, p: 'Participant') -> 'None'",
    "Component.spawn": "(self, coro: 'Coroutine[Any, Any, T]', *, key: 'str | None' = None, supersede: 'bool' = False) -> 'asyncio.Task[T]'",
}


@pytest.mark.parametrize("dotted", sorted(AGENT_PARTICIPANT_SIGNATURES))
def test_agent_participant_stub_signature_is_frozen(dotted: str) -> None:
    assert str(inspect.signature(getattr(Component, dotted.split(".")[1]))) == AGENT_PARTICIPANT_SIGNATURES[dotted]


def test_agent_spawn_mirrors_participant_spawn() -> None:
    """Same parameters, kinds and defaults as ``Participant.spawn`` (less ``detached``: no turn to detach from)."""
    assert _shape(Component.spawn) == [s for s in _shape(Participant.spawn) if s[0] != "detached"]
    assert (
        inspect.signature(Component.spawn).return_annotation == inspect.signature(Participant.spawn).return_annotation
    )
    assert not inspect.iscoroutinefunction(Component.spawn)


async def test_agent_enrollment_hooks_are_noops_and_spawn_needs_a_registered_agent() -> None:
    agent = Component()
    for hook in (agent.on_participant_enrolled, agent.on_participant_unenrolled):
        assert await hook(None) is None  # type: ignore[arg-type]
    coro = asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="not added to an app"):
        agent.spawn(coro)
    assert Participant("p").connected is False


def test_participant_connected_is_a_read_only_bool_property() -> None:
    prop = inspect.getattr_static(Participant, "connected")
    assert isinstance(prop, property)
    assert prop.fset is None
    assert inspect.signature(prop.fget).return_annotation == "bool"  # type: ignore[arg-type]


def test_connection_is_a_frozen_dataclass_of_two_strings() -> None:
    assert dataclasses.is_dataclass(enrollment.Connection)
    assert [(f.name, f.type) for f in dataclasses.fields(enrollment.Connection)] == [
        ("identity", "str"),
        ("session_id", "str"),
    ]
    c = enrollment.Connection("alice", "s1")
    assert (c.identity, c.session_id) == ("alice", "s1")
    assert c == enrollment.Connection(identity="alice", session_id="s1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.identity = "bob"  # type: ignore[misc]
    with pytest.raises(TypeError):
        enrollment.Connection("alice")  # type: ignore[call-arg]


def test_enrollment_protocol_members_are_read_only() -> None:
    assert set(enrollment.Enrollment.__protocol_attrs__) == {"admit", "keep_for_s"}
    assert all(isinstance(inspect.getattr_static(enrollment.Enrollment, n), property) for n in ("admit", "keep_for_s"))

    class Policy:
        admit = None
        keep_for_s = 5.0

    class NoKeep:
        admit = None

    assert isinstance(Policy(), enrollment.Enrollment)
    assert not isinstance(NoKeep(), enrollment.Enrollment)


def test_policies_are_exported_and_derive_from_nothing_special() -> None:
    for name in ("PerConnection", "Persistent"):
        assert name in xr_ai_app.__all__
        assert getattr(xr_ai_app, name) is getattr(enrollment, name)
    assert not issubclass(enrollment.PerConnection, Component)
    assert not issubclass(enrollment.Persistent, Component)


def test_participant_topics_and_messages() -> None:
    expected = {
        "PARTICIPANT_CONNECTED": ("xr_ai_app.participant_connected", topics.ParticipantConnected),
        "PARTICIPANT_DISCONNECTED": ("xr_ai_app.participant_disconnected", topics.ParticipantDisconnected),
    }
    for const, (name, message) in expected.items():
        topic = getattr(topics, const)
        assert (topic.name, topic.message_type) == (name, message)
        assert const in topics.__all__
        assert issubclass(message, BaseModel)
        assert message.model_config["extra"] == "forbid"
        assert message.__name__ in topics.__all__
        assert message.__name__ not in xr_ai_app.__all__  # the top level stays small (reachable from the topics module)
    for message in (topics.ParticipantConnected, topics.ParticipantDisconnected):
        assert list(message.model_fields) == ["connection"]
        conn = enrollment.Connection("alice", "s1")
        assert message(connection=conn).connection == conn
        assert message.model_validate({"connection": {"identity": "alice", "session_id": "s1"}}).connection == conn
        with pytest.raises(ValidationError):
            message()


# --- the old forms are gone -----------------------------------------------------------------------------------


def test_from_config_takes_a_required_keyword_enrollment() -> None:
    params = inspect.signature(App.from_config).parameters
    assert params["enrollment"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["enrollment"].default is inspect.Parameter.empty
    assert [n for n in params if n != "cls"] == ["path", "enrollment", "profile"]


def test_old_hooks_are_removed() -> None:
    assert not hasattr(Component, "on_participant_joined")
    assert not hasattr(Component, "on_participant_left")


def test_old_presence_topics_are_removed() -> None:
    assert not hasattr(topics, "PARTICIPANT_JOINED")
    assert not hasattr(topics, "PARTICIPANT_LEFT")
    assert "PARTICIPANT_JOINED" not in topics.__all__
    assert "PARTICIPANT_LEFT" not in topics.__all__


def test_old_presence_messages_are_removed_from_xr_ai() -> None:
    assert not hasattr(xr_ai_app, "ParticipantJoined")
    assert not hasattr(xr_ai_app, "ParticipantLeft")
    assert not hasattr(topics, "ParticipantJoined")
    assert hasattr(xr_ai_interaction, "ParticipantJoined")  # the package itself is untouched (additive-only rule)


def test_hub_signature_is_frozen_and_it_is_a_keyword_only_agent_with_no_required_wiring() -> None:
    from xr_ai_app._base.hub import Hub

    assert str(inspect.signature(Hub.__init__)) == (
        "(self, *, min_interval_s: 'float' = 0.5, track_id: 'str | None' = None, fetch_timeout_s: 'float' = 2.0, "
        "clock: \"Literal['pts', 'receive']\" = 'pts') -> 'None'"
    )
    assert issubclass(Hub, Component)
    Hub()  # constructing opens nothing: the endpoint is built in on_start
