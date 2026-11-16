# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
device_io_hub.ipc — extensible IPC layer for DeviceIOHub.

Endpoints
---------
ConnectorEndpoint   — producer (LiveKit connector process)
HubEndpoint         — server  (DeviceIOHub process)
ProcessorEndpoint   — subscriber + publisher (agents, analytics, downstream processors)

Agent code should import from `xr_ai_hub` directly rather than this module —
it avoids pulling in the full DeviceIOHub dependency tree.

Extensibility
-------------
Register new message types at import time:

    from device_io_hub.ipc import register_encoder, register_decoder, MsgType
    from enum import IntEnum

    class MyMsgType(IntEnum):
        MY_MSG = 10          # pick an ID outside 1-9 (built-ins)

    register_encoder(MyMsgType.MY_MSG, lambda m: [m.field_a, m.field_b])
    register_decoder(MyMsgType.MY_MSG, lambda p: MyMsg(p[0], p[1]))
"""

# Agent-facing types and endpoint — convenience re-exports from xr_ai_hub.
from xr_ai_hub import (
    AudioChunk,
    FrameData,
    PixelFormat,
    ProcessorEndpoint,
    Subscribe,
    decode,
    encode,
    register_decoder,
    register_encoder,
)
from xr_ai_hub._processor import AGENT_STATUS_TOPIC
from xr_ai_hub._shm import ShmRingBuffer, SlotView
from xr_ai_hub._types import (
    ConnectorRegistration,
    ControlMessage,
    DataMessage,
    FileMessage,
    FrameRequest,
    FrameSignal,
    MsgType,
    ParticipantEvent,
    ReturnAudioFlush,
    RosterRequest,
)

# Server-side endpoints — only available when device-io-hub is installed.
from ._connector import ConnectorEndpoint
from ._hub import (
    TOPIC_AUDIO,
    TOPIC_CONTROL,
    TOPIC_DATA,
    TOPIC_FILE,
    TOPIC_RETURN_AUDIO,
    TOPIC_RETURN_AUDIO_FLUSH,
    TOPIC_RETURN_DATA,
    TOPIC_VIDEO,
    TOPIC_VIDEO_DATA,
    HubEndpoint,
)

__all__ = [
    # endpoints
    "ConnectorEndpoint",
    "HubEndpoint",
    "ProcessorEndpoint",
    "Subscribe",
    # shared memory
    "ShmRingBuffer",
    "SlotView",
    # codec extension points
    "encode",
    "decode",
    "register_encoder",
    "register_decoder",
    # data types
    "AudioChunk",
    "ConnectorRegistration",
    "ControlMessage",
    "DataMessage",
    "FileMessage",
    "FrameData",
    "FrameRequest",
    "FrameSignal",
    "MsgType",
    "ParticipantEvent",
    "PixelFormat",
    "ReturnAudioFlush",
    "RosterRequest",
    # well-known topic prefixes
    "TOPIC_VIDEO",
    "TOPIC_VIDEO_DATA",
    "TOPIC_AUDIO",
    "TOPIC_DATA",
    "TOPIC_FILE",
    "TOPIC_CONTROL",
    "TOPIC_RETURN_AUDIO",
    "TOPIC_RETURN_AUDIO_FLUSH",
    "TOPIC_RETURN_DATA",
    # internal SDK channel topic
    "AGENT_STATUS_TOPIC",
]
