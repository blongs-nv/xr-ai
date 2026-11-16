# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness: the app reports ready only once the model servers behind its roles answer.

``App`` records one probe per model client it builds (every role the app binds, and speech out); before the voice
pipeline starts, or the ready file is written, it waits until every probe passes, polling like the voice package's
own service wait. A vision model is not only asked whether it is up: it answers one small image request, so the first
user question does not pay the multimodal warm-up (as upstream's simple-vlm sample does). Speaker speech recognition
is waited for by the voice pipeline itself when the voice gate requires it.
"""

from __future__ import annotations

import asyncio
import io
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

from loguru import logger

Probe = Callable[[], Awaitable[bool]]

POLL_S: Final = 5.0
"""Seconds between rounds of probes that have not passed yet."""

_WARMUP_SIZE: Final = (1280, 720)
_WARMUP_MAX_TOKENS: Final = 4
_WARMUP_TIMEOUT_S: Final = 120.0
_warmup_jpeg: bytes | None = None


def probe_for(category: str, client: Any) -> Probe:
    """The readiness probe of a model client: ``health()``, or a one-image warm-up for a vision model."""
    if category == "vlm" and hasattr(client, "stream_images"):
        return lambda: _warm_vlm(client)
    return lambda: client.health()


async def _warm_vlm(client: Any) -> bool:
    global _warmup_jpeg  # noqa: PLW0603 - built once, on first use
    if _warmup_jpeg is None:
        from PIL import Image  # noqa: PLC0415 - Pillow comes with the frame tools

        buffer = io.BytesIO()
        with Image.new("RGB", _WARMUP_SIZE, color=(128, 128, 128)) as image:
            image.save(buffer, format="JPEG", quality=90)
        _warmup_jpeg = buffer.getvalue()
    started = time.monotonic()
    try:
        async for _ in client.stream_images(
            [_warmup_jpeg],
            "What is the dominant color?",
            system_prompt="Answer with one word.",
            max_tokens=_WARMUP_MAX_TOKENS,
            timeout=_WARMUP_TIMEOUT_S,
        ):
            pass
    except Exception:  # noqa: BLE001 - a failed warm-up means not ready yet; the next round retries
        logger.opt(exception=True).info("vision model warm-up failed; retrying")
        return False
    logger.info("vision model warmed up in {:.0f} ms", (time.monotonic() - started) * 1000)
    return True


async def _passes(name: str, probe: Probe) -> bool:
    try:
        return bool(await probe())
    except Exception:  # noqa: BLE001 - a probe that raises has not passed
        logger.opt(exception=True).debug("probe {} raised", name)
        return False


async def wait_until_ready(probes: Mapping[str, Probe]) -> None:
    """Return once every probe has passed once; probes that have not are retried every :data:`POLL_S` seconds."""
    pending = dict(probes)
    while pending:
        for name, probe in list(pending.items()):
            if await _passes(name, probe):
                logger.info("{} ready", name)
                del pending[name]
        if pending:
            logger.info("waiting for model servers: {}", ", ".join(sorted(pending)))
            await asyncio.sleep(POLL_S)
