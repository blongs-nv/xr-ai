# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A minimal OpenAI-compatible chat client that reports latency and token usage."""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from xr_ai_eval.profiles import Endpoint


@dataclass(frozen=True)
class ChatResult:
    """One completion: text, wall time and reported usage (``None`` when not reported)."""

    text: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    replies: tuple[str, ...] = ()


def chat_url(base_url: str) -> str:
    """Join a base URL and the chat path; local profiles omit ``/v1``, hosted ones include it."""
    base = base_url.rstrip("/")
    return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"


def transcription_url(base_url: str) -> str:
    """Join a base URL and the transcription path (same ``/v1`` rule as ``chat_url``)."""
    base = base_url.rstrip("/")
    return f"{base}/audio/transcriptions" if base.endswith("/v1") else f"{base}/v1/audio/transcriptions"


def _auth_headers(endpoint: Endpoint) -> dict[str, str]:
    if not endpoint.api_key_env:
        return {}
    key = os.environ.get(endpoint.api_key_env)
    if not key:
        raise ValueError(f"environment variable {endpoint.api_key_env} is not set")
    return {"Authorization": f"Bearer {key}"}


def _image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_messages(
    text: str,
    frames: tuple[Path, ...],
    context: tuple[dict[str, str], ...],
    system: str | None = None,
) -> list[dict[str, Any]]:
    """Build chat messages: optional system prompt, prior context, then the user turn with any frames.

    Frames are sent in the order given (earlier first), after the text.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    messages += [{"role": m["role"], "content": m["text"]} for m in context]
    if frames:
        content: Any = [{"type": "text", "text": text}, *(_image_part(f) for f in frames)]
    else:
        content = text
    messages.append({"role": "user", "content": content})
    return messages


class ChatClient:
    """Calls ``/v1/chat/completions`` on an endpoint."""

    def __init__(self, timeout_s: float = 120.0, max_tokens: int = 512) -> None:
        self._http = httpx.AsyncClient(timeout=timeout_s)
        self._max_tokens = max_tokens

    async def chat(self, endpoint: Endpoint, messages: list[dict[str, Any]]) -> ChatResult:
        """Send one request. Raises ``httpx.HTTPError`` or ``ValueError`` on failure."""
        headers = _auth_headers(endpoint)
        body = {"model": endpoint.model, "messages": messages, "temperature": 0.0, "max_tokens": self._max_tokens}
        started = time.perf_counter()
        response = await self._http.post(chat_url(endpoint.base_url), json=body, headers=headers)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        payload = response.json()
        try:
            text = payload["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError(f"unexpected response shape: {str(payload)[:200]}") from exc
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return ChatResult(text, latency_ms, _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens")))

    async def transcribe(self, endpoint: Endpoint, audio: Path) -> ChatResult:
        """Post ``audio`` to ``/v1/audio/transcriptions`` (multipart) and return the ``text`` field."""
        headers = _auth_headers(endpoint)
        mime = mimetypes.guess_type(audio.name)[0] or "audio/wav"
        started = time.perf_counter()
        response = await self._http.post(
            transcription_url(endpoint.base_url),
            files={"file": (audio.name, audio.read_bytes(), mime)},
            data={"model": endpoint.model, "response_format": "json"},
            headers=headers,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        try:
            text = str(response.json()["text"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"unexpected transcription response: {response.text[:200]}") from exc
        return ChatResult(text, latency_ms, None, None)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
