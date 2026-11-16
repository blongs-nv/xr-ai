# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr_ai_hub — lightweight agent-side SDK for DeviceIOHub.

Agents only need this package (pyzmq + msgpack). The hub implementation and
its LiveKit, FastAPI, and uvicorn dependencies are not included.

Typical usage::

    from xr_ai_hub import ProcessorEndpoint, DataMessage, FrameSignal

    ep = ProcessorEndpoint(sub_addr="ipc:///tmp/xr_hub_pub",
                           push_addr="ipc:///tmp/xr_hub_in")
    ep.on_frame(my_frame_handler)
    ep.on_data(my_data_handler)
    await ep.run()
"""

from ._codec import decode, encode, register_decoder, register_encoder
from ._live_frames import FrameUnavailable, LiveFrameSource
from ._processor import ProcessorEndpoint, Subscribe
from ._types import AudioChunk, DataMessage, FrameData, FrameSignal, PixelFormat

__all__ = [
    # endpoint
    "ProcessorEndpoint",
    "Subscribe",
    # codec extension points
    "encode",
    "decode",
    "register_encoder",
    "register_decoder",
    # data types
    "AudioChunk",
    "DataMessage",
    "FrameData",
    "FrameUnavailable",
    "FrameSignal",
    "LiveFrameSource",
    "PixelFormat",
]

# Names that left the public surface: still importable, each with a DeprecationWarning that names the
# replacement. `__all__` no longer lists them (docs/source/reference/migrations.md has the before and after).
_WIRE = (
    "internal wire type; apps use xr_ai_next (Hub, Participant.send_data, Frame). "
    "Hub plumbing imports it from xr_ai_hub._types"
)
_SHM = (
    "internal ring buffer; apps read frames through Participant.frames. Hub plumbing imports it from xr_ai_hub._shm"
)
_STATUS = (
    "internal wire topic; apps report status with Participant.status(). "
    "Hub plumbing imports it from xr_ai_hub._processor"
)
_DEPRECATED: dict[str, tuple[str, str]] = {
    "AgentPresence": ("._types", _WIRE),
    "ConnectorRegistration": ("._types", _WIRE),
    "ControlMessage": ("._types", _WIRE),
    "FileMessage": ("._types", _WIRE),
    "FrameRequest": ("._types", _WIRE),
    "MsgType": ("._types", _WIRE),
    "ParticipantEvent": ("._types", _WIRE),
    "ReturnAudioFlush": ("._types", _WIRE),
    "RosterRequest": ("._types", _WIRE),
    "SubscriptionProbe": ("._types", _WIRE),
    "AGENT_STATUS_TOPIC": ("._processor", _STATUS),
    "ShmRingBuffer": ("._shm", _SHM),
    "SlotView": ("._shm", _SHM),
}


def __getattr__(name: str) -> object:
    entry = _DEPRECATED.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import warnings  # noqa: PLC0415

    module, replacement = entry
    # `from pkg import name` first probes with hasattr(); importlib calls it from a `<frozen importlib...` frame. Skip
    # the warning for that probe only: the real lookup that follows warns, so an import warns once, not twice.
    if not sys._getframe(1).f_code.co_filename.startswith("<frozen importlib"):
        warnings.warn(f"{__name__}.{name} is deprecated: {replacement}", DeprecationWarning, stacklevel=2)
    return getattr(importlib.import_module(module, __package__), name)


def __dir__() -> list[str]:
    return sorted([*globals(), *_DEPRECATED])
