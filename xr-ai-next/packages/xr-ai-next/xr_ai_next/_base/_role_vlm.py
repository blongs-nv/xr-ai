# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Role.as_vlm()``: a ``VLMService`` over a role (mirrors ``Role.as_llm``)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from xr_ai_models import Capabilities, ChatResponse, ImageInput, VideoInput

from ._bound import BoundRole
from .profiles import Role


class _RoleVLM:
    """A ``VLMService`` over a role, resolved when first used.

    A bound role goes to its client under the role's concurrency limit, with every option passed through. Any
    other role (the test kit's fake roles) is driven through ``ask``: the images become frames, and only the
    system prompt is kept (``ask_video`` is not available there).
    """

    def __init__(self, resolve: Callable[[], Role]) -> None:
        self._resolve = resolve

    @property
    def capabilities(self) -> Capabilities:
        role = self._resolve()
        client = getattr(role, "_client", None)
        return getattr(client, "capabilities", None) or Capabilities(vision=True)

    @staticmethod
    def _limit(role: Role) -> Any:
        limiter = getattr(role, "_limiter", None)
        return limiter if limiter is not None else contextlib.nullcontext()

    async def ask_image(self, image: ImageInput, question: str, **options: Any) -> ChatResponse:
        role = self._resolve()
        if isinstance(role, BoundRole):
            async with self._limit(role):
                return await role._client.ask_image(image, question, **options)
        return await self._ask(role, [image], question, options)

    async def ask_images(self, images: Sequence[ImageInput], question: str, **options: Any) -> ChatResponse:
        role = self._resolve()
        if isinstance(role, BoundRole):
            async with self._limit(role):
                return await role._client.ask_images(images, question, **options)
        return await self._ask(role, images, question, options)

    async def ask_video(self, video: VideoInput, question: str, **options: Any) -> ChatResponse:
        role = self._resolve()
        if isinstance(role, BoundRole):
            async with self._limit(role):
                return await role._client.ask_video(video, question, **options)
        raise NotImplementedError("ask_video needs a role bound to a real client; a test-kit role cannot answer it")

    @staticmethod
    async def _ask(role: Role, images: Sequence[ImageInput], question: str, options: Mapping[str, Any]) -> ChatResponse:
        from .frames import Frame  # noqa: PLC0415 - ``frames`` imports this module's package

        frames = [Frame(image=await _bytes(i), timestamp_us=0, width=0, height=0) for i in images]
        system = options.get("system_prompt") or None
        reply = await role.ask(
            frames,
            question,
            system=system,
            max_tokens=options.get("max_tokens"),
            temperature=options.get("temperature"),
            timeout_s=options.get("timeout"),
        )
        return ChatResponse(str(reply), None, None, "stop", {})


async def _bytes(image: ImageInput) -> bytes:
    return image if isinstance(image, bytes) else await asyncio.to_thread(Path(image).read_bytes)
