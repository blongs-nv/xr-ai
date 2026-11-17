# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The XR-AI app SDK, the core: the primitives every voice/video-first app is built from.

The core is what cannot be built from other public pieces (it needs the runtime, the hub or the model binding).
Top-level names are for every app. Opt-in modules: ``xr_ai_app.trace`` (events and sinks), ``.clock`` (time that
stays right on the devkit's virtual clock), ``.context`` (``render`` and the channel protocol), ``.tools`` and
``.llm`` (tools as data and the raw model interface, for tool loops and adapters). Tools for developers: ``.devkit``
(run, exercise and serve an app without a headset) and ``.testing``. Patterns composed from these (Conversation with
its tool loop, Router, Delegate) are the kit, ``xr-ai-kit`` (``xr_ai_kit``), with libraries for one kind of app or
one framework beside them (``xr_ai_kit.sop`` for procedures, which also holds watching the camera over time, and
``xr_ai_kit.langchain``). Kits use only this package's public names, and this package never imports them.

======================  =================================================================
App and lifecycle       App, Component, PerConnection, Persistent
Participants            Participant, Session, Turn (``p.state``, ``p.session.state``,
                        ``turn.state``), current_turn
Device and voice        Hub (the device connection: who is here, and their camera), Frame, FrameStream
                        (``p.frames``), Voice
Outputs                 Sink, TextSink (and Voice), output_started (whether the current turn
                        has said anything yet)
Models                  Role, Decision, StructuredOutputError, tool (routing is ``role.decide``)
Context                 ContextLog, ContextValue
======================  =================================================================
"""

from __future__ import annotations

from ._base.app import App
from ._base.component import Component
from ._base.enrollment import PerConnection, Persistent
from ._base.frames import Frame, FrameStream
from ._base.hub import Hub
from ._base.participant import Participant, Session, Turn, current_turn
from ._base.profiles import Decision, Role, StructuredOutputError
from ._base.tools import tool
from .context import ContextLog, ContextValue
from .output import Sink, TextSink, output_started
from .voice import Voice

__all__ = [
    "App",
    "Component",
    "ContextLog",
    "ContextValue",
    "Decision",
    "Frame",
    "FrameStream",
    "Hub",
    "Participant",
    "PerConnection",
    "Persistent",
    "Role",
    "Session",
    "Sink",
    "StructuredOutputError",
    "TextSink",
    "Turn",
    "Voice",
    "current_turn",
    "output_started",
    "tool",
]
