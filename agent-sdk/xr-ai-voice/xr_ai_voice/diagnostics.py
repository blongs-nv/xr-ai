# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped voice health and output-handoff diagnostics."""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger

__all__ = ["observe"]

_OBSERVER: contextvars.ContextVar[Callable[[dict[str, Any]], None] | None] = contextvars.ContextVar(
    "xr_ai_speech_observer", default=None
)


@contextmanager
def observe(callback: Callable[[dict[str, Any]], None]) -> Iterator[None]:
    """Send diagnostics from speech tasks created in this context to ``callback``.

    Events carry ``kind`` and ``participant``; queue events add ``seconds`` and
    speaker events add ``event``. Response events include completed assistant text
    handed to TTS and request metadata; input transcripts and audio are omitted.
    Runtime streams closed by idle timeout, eviction or shutdown without a producer
    terminator emit no response event. The callback must not block.
    """
    token = _OBSERVER.set(callback)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def _emit(kind: str, participant: str, **data: Any) -> None:
    callback = _OBSERVER.get()
    if callback is not None:
        try:
            callback({"kind": kind, "participant": participant, **data})
        except Exception:  # noqa: BLE001 - diagnostics must not interrupt speech
            logger.opt(exception=True).debug("speech diagnostic observer failed")
