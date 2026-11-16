# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Role.as_vlm()`` is the public ``VLMService`` over a role (bound and test-kit roles)."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from xr_ai_models import Capabilities, ChatResponse
from xr_ai_next._base.app import App, AppConfig, VoiceConfig
from xr_ai_next._base.profiles import Role
from xr_ai_next._base.testing import FakeModels, test_session


class _Client:
    capabilities = Capabilities(vision=True, video=True)

    def __init__(self) -> None:
        self.seen: list[tuple[str, Any, str, dict[str, Any]]] = []
        self.block = asyncio.Event()

    async def ask_image(self, image: Any, question: str, **options: Any) -> ChatResponse:
        self.seen.append(("image", image, question, options))
        if options.get("max_tokens") == 1:
            await self.block.wait()
        return ChatResponse("one", None, None, "stop", {})

    async def ask_images(self, images: Any, question: str, **options: Any) -> ChatResponse:
        self.seen.append(("images", list(images), question, options))
        return ChatResponse("many", None, None, "stop", {})

    async def ask_video(self, video: Any, question: str, **options: Any) -> ChatResponse:
        self.seen.append(("video", video, question, options))
        return ChatResponse("clip", None, None, "stop", {})


def _bound(client: _Client, limiter: asyncio.Semaphore | None = None) -> Role:
    return Role.bind("perceive", entry="vlm", category="vlm", client=client, limiter=limiter)


async def test_a_bound_role_goes_to_its_client_with_every_option() -> None:
    client = _Client()
    vlm = _bound(client).as_vlm()
    assert vlm.capabilities.video
    assert (await vlm.ask_image(b"x", "q", system_prompt="s", max_tokens=5)).content == "one"
    assert (await vlm.ask_images([b"x", b"y"], "q")).content == "many"
    assert (await vlm.ask_video(b"v", "q")).content == "clip"
    assert [s[0] for s in client.seen] == ["image", "images", "video"]
    assert client.seen[0][3] == {"system_prompt": "s", "max_tokens": 5}


async def test_cancelling_a_bound_call_releases_the_concurrency_limit() -> None:
    client = _Client()
    limiter = asyncio.Semaphore(1)
    vlm = _bound(client, limiter).as_vlm()
    call = asyncio.ensure_future(vlm.ask_image(b"x", "q", max_tokens=1))
    await asyncio.sleep(0.01)
    assert limiter.locked()
    call.cancel()
    await asyncio.gather(call, return_exceptions=True)
    assert not limiter.locked()


async def test_a_test_kit_role_answers_through_ask_and_has_no_video() -> None:
    app = App(AppConfig(config_dir=Path("."), voice=VoiceConfig(enabled=False)))
    models = FakeModels({"perceive": ["a red kettle", "two cups"]})
    async with test_session(app, models=models):
        vlm = app.role("perceive").as_vlm()
        assert (await vlm.ask_image(b"png", "what is it?", system_prompt="be brief")).content == "a red kettle"
        assert (await vlm.ask_images([b"a", b"b"], "how many?")).content == "two cups"
        with pytest.raises(NotImplementedError):
            await vlm.ask_video(b"v", "q")
    first = models.calls("perceive")[0]
    assert first.system == "be brief" and first.prompt == "what is it?"
