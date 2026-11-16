# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Role.as_llm()``: an ``LLMService`` over a role (mirrored by ``Role.as_vlm`` in :mod:`._role_vlm`)."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any

from xr_ai_models import ChatMessage, ChatResponse, ChatStreamEvent
from xr_ai_models import ToolCall as _ModelToolCall

from ._bound import BoundRole

if TYPE_CHECKING:
    from .profiles import Role


def _opts(**options: Any) -> dict[str, Any]:
    return {key: value for key, value in options.items() if value is not None}


def _render(messages: Sequence[ChatMessage]) -> tuple[str, str | None]:
    """Prompt text and system text of a transcript, for roles that take one prompt (fakes, vision)."""
    system: list[str] = []
    parts: list[str] = []
    names: dict[str, str] = {}
    for message in messages:
        content = message.content if isinstance(message.content, str) else str(message.content)
        if message.role == "system":
            system.append(content)
        elif message.role == "tool":
            parts.append(f"Tool {names.get(message.tool_call_id or '', '?')} returned: {content}")
        else:
            if content:
                parts.append(content)
            for call in message.tool_calls or ():
                names[call.id] = call.name
                parts.append(f"You called tool {call.name} with {call.arguments}")
    return "\n\n".join(parts), ("\n\n".join(system) or None)


def _fake_tool_call(reply: Any, index: int) -> _ModelToolCall | None:
    from .testing import ToolCall  # noqa: PLC0415 - the test kit imports this package

    if not isinstance(reply, ToolCall):
        return None
    return _ModelToolCall(id=f"call_{index}", name=reply.name, arguments=json.dumps(dict(reply.arguments)))


class _RoleLLM:
    """An ``LLMService`` over a role, resolved when first used.

    A bound role goes to its client (tools, streaming, the role's concurrency limit). Any other role
    (the test kit's fake roles) is driven through ``ask``/``stream`` with the transcript rendered as
    one prompt; ``ask`` is the only call that can return a scripted tool call.
    """

    def __init__(self, resolve: Callable[[], Role]) -> None:
        self._resolve = resolve
        self._calls = 0

    @staticmethod
    def _limit(role: Role) -> Any:
        limiter = getattr(role, "_limiter", None)
        return limiter if limiter is not None else contextlib.nullcontext()

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Any = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
        **_ignored: Any,
    ) -> ChatResponse:
        role = self._resolve()
        if isinstance(role, BoundRole):
            role._require_text_model("as_llm")  # a decision or speech role has no chat
            async with self._limit(role):
                return await role._client.chat(
                    messages,
                    **_opts(
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        enable_thinking=enable_thinking,
                    ),
                )
        prompt, system = _render(messages)
        reply = await role.ask(
            None, prompt, system=system, thinking=enable_thinking, max_tokens=max_tokens, temperature=temperature
        )
        call = _fake_tool_call(reply, self._calls)
        self._calls += 1
        if call is not None:
            return ChatResponse("", None, [call], "tool_calls", {})
        return ChatResponse(str(reply), None, None, "stop", {})

    async def stream_events(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Any = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
        **_ignored: Any,
    ) -> AsyncIterator[ChatStreamEvent]:
        role = self._resolve()
        if isinstance(role, BoundRole):
            role._require_text_model("as_llm")
            async with self._limit(role):
                events = role._client.stream_events(
                    messages,
                    **_opts(
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        enable_thinking=enable_thinking,
                    ),
                )
                try:
                    async for event in events:
                        yield event
                finally:
                    await _aclose(events)
            return
        if tools:  # a scripted tool call can only come back from ``ask``
            response = await self.chat(
                messages, tools=tools, max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking
            )
            for call in response.tool_calls or ():
                yield ChatStreamEvent(kind="tool_call", tool_call=call)
            if response.content:
                yield ChatStreamEvent(kind="text", text=response.content)
            yield ChatStreamEvent(kind="finish", finish_reason=response.finish_reason)
            return
        prompt, system = _render(messages)
        stream = role.stream(
            None, prompt, system=system, thinking=enable_thinking, max_tokens=max_tokens, temperature=temperature
        )
        try:
            async for chunk in stream:
                if chunk:
                    yield ChatStreamEvent(kind="text", text=chunk)
        finally:
            await _aclose(stream)
        yield ChatStreamEvent(kind="finish", finish_reason="stop")


async def _aclose(iterator: Any) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()
