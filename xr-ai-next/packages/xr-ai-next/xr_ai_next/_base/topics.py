# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard topics shared by every app.

The query and output payload types are the transport-neutral ``xr_ai_interaction`` message types. The connection
events have exactly one publisher, the ``Hub`` component (R8.4); what they mean for participants is the app's
enrollment setting. The participant id is in the runtime context of every one of them.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict
from xr_ai_interaction import Interrupted, ResponseOutput, UserQuery
from xr_ai_runtime import Topic

from .enrollment import Connection


class ParticipantConnected(BaseModel):
    """A device connected (published by the ``Hub`` component); the participant id is in the runtime context."""

    model_config = ConfigDict(extra="forbid")

    connection: Connection


class ParticipantDisconnected(BaseModel):
    """A device disconnected (published by the ``Hub`` component)."""

    model_config = ConfigDict(extra="forbid")

    connection: Connection


USER_QUERY: Final = Topic("xr_ai_next.user_query", UserQuery)
"""An accepted user query."""

PARTICIPANT_CONNECTED: Final = Topic("xr_ai_next.participant_connected", ParticipantConnected)
"""A device connected; one publisher, the ``Hub`` component (R8.4)."""

PARTICIPANT_DISCONNECTED: Final = Topic("xr_ai_next.participant_disconnected", ParticipantDisconnected)
"""A device disconnected; one publisher, the ``Hub`` component."""

INTERRUPTED: Final = Topic("xr_ai_next.interrupted", Interrupted)
"""The participant's current response was interrupted."""

OUTPUT: Final = Topic("xr_ai_next.output", ResponseOutput)
"""Spoken or displayed output for a participant."""

__all__ = [
    "INTERRUPTED",
    "OUTPUT",
    "PARTICIPANT_CONNECTED",
    "PARTICIPANT_DISCONNECTED",
    "USER_QUERY",
    "Interrupted",
    "ParticipantConnected",
    "ParticipantDisconnected",
    "ResponseOutput",
]
