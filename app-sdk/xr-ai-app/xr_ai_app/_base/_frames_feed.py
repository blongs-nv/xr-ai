# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real hub frames into ``p.frames``; the machinery of the ``Hub`` component.

A :class:`HubFrameFeed` wraps one ``ProcessorEndpoint`` (no signal handlers, no readiness vote) that the ``Hub``
component runs as component-lifetime work and closes in its stop hook. Each enrolled participant gets one pump task
in its scope (key ``frames``); it fetches pixels only while the participant is connected.

Throttle: the hub sends a ``FrameSignal`` (metadata only) at the camera rate, 30 per second or more.
Pixels are copied only when the pump asks, so the pump asks **at most once every ``min_interval_s``**
(default 0.5 s) whatever the signal rate, and always for the freshest signal seen at that moment. It
sleeps when no signal arrives, so an idle camera costs nothing. The newest frame is therefore up to
``min_interval_s`` plus one fetch older than the camera's newest. ``FrameStream`` keeps one frame.

Track: ``Frame`` has no track id (GAP-sop-06). With ``track_id=None`` the signal with the largest
``pts_us`` across the participant's tracks wins; pass ``track_id`` to follow one track only.

Clock skew: ``Frame.timestamp_us`` is compared by ``FrameStream.current(max_age_s=...)`` with this
process's wall clock. The hub supplies no receive time, only ``pts_us``, which is the sender's clock
(the same assumption ``xr_ai_hub.LiveFrameSource`` makes). With ``clock="pts"`` (default) ``pts_us`` is
used when it is within a day of the local clock and the local receive time otherwise (a stream-relative
pts would make every frame look stale). With ``clock="receive"`` the local time at which the pixels
arrived is always used: immune to device skew, late by the signal-to-fetch delay.

On-demand pictures: each participant's ``p.frames`` can ask the client for one still image (upstream's
``_ClientImageCaptureSource``; the reply arrives on ``data.<pid>.camera.capture.response``, hence ``Subscribe.DATA``).
A client in the web client's on-demand camera mode turns its camera on for one frame; a live-streaming client
rejects the request and the stream answers instead.
"""

from __future__ import annotations

import asyncio
import io
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from .frames import Frame

if TYPE_CHECKING:
    from xr_ai_hub import FrameData
    from xr_ai_hub._types import FrameSignal

    from .participant import Participant

from xr_ai_hub.isolation import DEFAULT_ENDPOINTS

_HUB_PUB, _HUB_PUSH = DEFAULT_ENDPOINTS
_PTS_PLAUSIBLE_US = 86_400 * 1_000_000
_PICTURE_TIMEOUT_S = 10.0  # the client's own bound; ``FrameStream.current`` cancels earlier at its ``timeout_s``


def encode_frame(data: FrameData) -> tuple[bytes, str]:
    """Encode raw hub pixels (I420, NV12, RGB24, RGBA or BGRA) as JPEG; returns ``(bytes, mime_type)``.

    Raises ``ValueError`` for an unsupported pixel format.
    """
    from xr_ai_tools._pixels import (  # noqa: PLC0415 - numpy and Pillow load on first call
        encode_image_bytes,
        frame_to_pil,
    )

    return encode_image_bytes(frame_to_pil(data)), "image/jpeg"


def hub_endpoint() -> Any:
    """Build a ``ProcessorEndpoint`` at the hub's standard local sockets (video, and data for picture replies).

    The endpoint exits the process when a callback raises, so every callback registered on it contains its own
    errors: ``HubFrameFeed._on_signal`` itself, ``Hub``'s participant callback through its component scope, and the
    two of upstream's capture source, which only resolve its own futures (a reply's JSON is parsed under a guard).
    """
    from xr_ai_hub import ProcessorEndpoint, Subscribe  # noqa: PLC0415

    return ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH, filter=Subscribe.VIDEO | Subscribe.DATA)


def client_pictures(endpoint: Any) -> Callable[[str], Awaitable[Frame | None]]:
    """Return ``take(participant_id)``: one still image from that participant's client, ``None`` when unavailable.

    Wraps upstream's ``_ClientImageCaptureSource`` on ``endpoint`` (which must subscribe to data). Cancelling ``take``
    cancels the request on the client. Call before registering other participant callbacks on a test double.
    """
    from xr_ai_hub._image_capture import (  # noqa: PLC0415
        _ClientImageCaptureSource,
        _ImageCaptureUnavailable,
    )

    source = _ClientImageCaptureSource(endpoint, timeout_s=_PICTURE_TIMEOUT_S)

    async def take(participant_id: str) -> Frame | None:
        try:
            image = await source.capture(participant_id)
            width, height = await asyncio.to_thread(_image_size, image.data)
        except _ImageCaptureUnavailable as exc:
            logger.info("no picture from {}: {}", participant_id, exc)
            return None
        except Exception:  # noqa: BLE001 - containment boundary (rule R4): a bad picture is no picture
            logger.exception("picture from {} failed", participant_id)
            return None
        now = time.time_ns() // 1000
        stamp = image.pts_us if abs(now - image.pts_us) < _PTS_PLAUSIBLE_US else now
        return Frame(image=image.data, timestamp_us=stamp, width=width, height=height, mime_type=image.mime_type)

    return take


def _image_size(data: bytes) -> tuple[int, int]:
    from PIL import Image  # noqa: PLC0415 - Pillow comes with the frame tools

    with Image.open(io.BytesIO(data)) as image:
        return image.size


class _Pump:
    def __init__(self) -> None:
        self.signals: dict[str, FrameSignal] = {}
        self.event = asyncio.Event()
        self.sequence = 0


class HubFrameFeed:
    """Owns a ``ProcessorEndpoint`` and feeds each participant's ``FrameStream`` from it."""

    def __init__(
        self,
        endpoint: Any,
        *,
        min_interval_s: float = 0.5,
        track_id: str | None = None,
        fetch_timeout_s: float = 2.0,
        clock: Literal["pts", "receive"] = "pts",
        encode: Callable[[Any], tuple[bytes, str]] = encode_frame,
        pictures: Callable[[Any], Callable[[str], Awaitable[Frame | None]]] = client_pictures,
    ) -> None:
        self._endpoint = endpoint
        self._take_picture = pictures(endpoint)
        self._min_interval_s = min_interval_s
        self._track_id = track_id
        self._fetch_timeout_s = fetch_timeout_s
        self._clock = clock
        self._encode = encode
        self._pumps: dict[str, _Pump] = {}
        endpoint.on_frame(self._on_signal)

    async def run(self) -> None:
        """Receive hub messages until cancelled (an app task)."""
        await self._endpoint.run()

    def close(self) -> None:
        """Stop and close the endpoint (stop sequence)."""
        self._endpoint.stop()
        self._endpoint.close()

    def attach(self, p: Participant) -> None:
        """Start the pump of ``p`` in its scope (it ends with the scope); ``p.frames`` may ask for a picture."""
        pump = self._pumps[p.id] = _Pump()
        p.frames._capture = lambda: self._take_picture(p.id)
        p._scope.spawn(self._pump(p, pump), key="frames")

    async def _on_signal(self, signal: FrameSignal) -> None:
        # The endpoint calls os._exit(1) when a callback raises: a malformed signal must not end the app.
        try:
            pump = self._pumps.get(signal.participant_id)
            if pump is None or (self._track_id is not None and signal.track_id != self._track_id):
                return
            pump.signals[signal.track_id] = signal
            pump.event.set()
        except Exception:  # noqa: BLE001 - containment boundary (rule R4): one bad signal, not the process
            logger.exception("frame signal failed: {!r}", signal)

    @staticmethod
    def _freshest(pump: _Pump) -> FrameSignal:
        return max(pump.signals.values(), key=lambda s: s.pts_us)

    def _timestamp_us(self, pts_us: int) -> int:
        now = time.time_ns() // 1000
        if self._clock == "pts" and abs(now - pts_us) < _PTS_PLAUSIBLE_US:
            return pts_us
        return now

    async def _pump(self, p: Participant, pump: _Pump) -> None:
        loop = asyncio.get_running_loop()
        last_fetch = float("-inf")
        try:
            while True:
                await pump.event.wait()
                if not p.connected:  # enrolled but nobody is connected: no pixels to fetch, no stale frame to keep
                    pump.signals.clear()
                    pump.event.clear()
                    continue
                wait = last_fetch + self._min_interval_s - loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                pump.event.clear()
                signal = self._freshest(pump)
                last_fetch = loop.time()
                try:
                    data = await self._endpoint.request_frame(signal, timeout=self._fetch_timeout_s)
                    if data is None:
                        continue
                    image, mime = await asyncio.to_thread(self._encode, data)
                except Exception:  # noqa: BLE001 - containment boundary (rule R4): one bad frame, not the feed
                    logger.exception("frame fetch failed for {}", p.id)
                    continue
                pump.sequence += 1
                frame = Frame(
                    image=image,
                    timestamp_us=self._timestamp_us(data.pts_us),
                    width=data.width,
                    height=data.height,
                    mime_type=mime,
                    sequence=pump.sequence,
                )
                if p._scope.admitting and p.connected:
                    p.frames._push(frame)
        finally:
            self._pumps.pop(p.id, None)
