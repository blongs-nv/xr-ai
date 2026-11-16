# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Role.as_llm()`` is the public tool-calling chat over a role (bound and test-kit roles)."""

import asyncio
from pathlib import Path
from typing import Any

from xr_ai_models import ChatMessage, ChatResponse, ChatStreamEvent
from xr_ai_next._base.app import App, AppConfig, VoiceConfig
from xr_ai_next._base.profiles import Role
from xr_ai_next._base.testing import FakeModels, ToolCall, test_session


class _Client:
    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []
        self.block = asyncio.Event()
        self.closed_streams = 0

    async def chat(self, messages: Any, **options: Any) -> ChatResponse:
        self.seen.append(options)
        if options.get("max_tokens") == 1:
            await self.block.wait()
        return ChatResponse("hi", None, None, "stop", {})

    async def stream_events(self, messages: Any, **options: Any) -> Any:
        try:
            yield ChatStreamEvent(kind="text", text="a")
            await self.block.wait()
            yield ChatStreamEvent(kind="finish", finish_reason="stop")
        finally:
            self.closed_streams += 1


def _bound(client: _Client, limiter: asyncio.Semaphore | None = None) -> Role:
    return Role.bind("answer", entry="llm", category="llm", client=client, limiter=limiter)


async def test_a_bound_role_chats_through_its_client_under_its_limit() -> None:
    client = _Client()
    limiter = asyncio.Semaphore(1)
    llm = _bound(client, limiter).as_llm()
    response = await llm.chat([ChatMessage(role="user", content="x")], max_tokens=7, temperature=0.0)
    assert response.content == "hi"
    assert client.seen[0]["max_tokens"] == 7 and client.seen[0]["temperature"] == 0.0
    assert not limiter.locked()


async def test_cancelling_a_bound_chat_or_stream_releases_the_limit_and_closes_the_stream() -> None:
    client = _Client()
    limiter = asyncio.Semaphore(1)
    llm = _bound(client, limiter).as_llm()
    chat = asyncio.ensure_future(llm.chat([ChatMessage(role="user", content="x")], max_tokens=1))
    await asyncio.sleep(0.01)
    assert limiter.locked()
    chat.cancel()
    await asyncio.gather(chat, return_exceptions=True)
    assert not limiter.locked()

    async def consume() -> None:
        async for _ in llm.stream_events([ChatMessage(role="user", content="x")]):
            pass

    stream = asyncio.ensure_future(consume())
    await asyncio.sleep(0.01)
    stream.cancel()
    await asyncio.gather(stream, return_exceptions=True)
    assert client.closed_streams == 1 and not limiter.locked()


async def test_a_test_kit_role_returns_scripted_tool_calls_from_chat() -> None:
    app = App(AppConfig(config_dir=Path("."), voice=VoiceConfig(enabled=False)))
    models = FakeModels({"answer": [ToolCall("lookup", {"q": "tea"}), "done"]})
    async with test_session(app, models=models):
        llm = app.role("answer").as_llm()
        first = await llm.chat([ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u")])
        assert [(c.name, c.arguments) for c in first.tool_calls or ()] == [("lookup", '{"q": "tea"}')]
        second = await llm.chat([ChatMessage(role="user", content="u")])
        assert second.content == "done" and not second.tool_calls
    assert models.calls("answer")[0].system == "s"
