# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context channels into LangChain messages, and LangChain/LangGraph message streams out as text to speak."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from xr_ai_next.context import ContextChannel, render

from ._chat import _text

__all__ = ["messages", "stream_text"]


async def messages(p: Any, *channels: ContextChannel, budget_bytes: int | None = None) -> list[BaseMessage]:
    """``channels`` as LangChain messages: ONE ``SystemMessage``, or ``[]`` when every channel is empty.

    The text is what ``Conversation`` puts in its system prompt: a ``# Context`` heading, then each non-empty channel
    under its name, in the given order (``budget_bytes`` caps each channel). One message rather than one per channel,
    because many chat templates accept a single system message; put it right after the agent's own prompt (the
    chat model merges consecutive system messages).
    """
    text = await render(p, channels, budget_bytes=budget_bytes)
    return [SystemMessage(content=f"# Context\n\n{text}")] if text else []


async def stream_text(stream: AsyncIterable[Any]) -> AsyncIterator[str]:
    """The AI text in a LangGraph or LangChain stream, fragment by fragment, for ``p.reply(..., to=voice)``.

    Accepts ``graph.astream(..., stream_mode="messages")`` (``(message, metadata)`` pairs), a list ``stream_mode``
    that includes ``"messages"`` (``(mode, payload)`` pairs; other modes are skipped), subgraph streams
    (``(namespace, ...)``), a chat model's ``astream`` (message chunks) and plain strings. Only AI messages count;
    a message or chunk carrying tool calls is skipped, as are tool, human and system messages and reasoning blocks.
    """
    async for item in stream:
        message = _message_of(item)
        if isinstance(message, str):
            if message:
                yield message
            continue
        if not isinstance(message, AIMessage):
            continue
        if message.tool_calls or getattr(message, "tool_call_chunks", None) or message.invalid_tool_calls:
            continue
        text = _text(message.content)
        if text:
            yield text


def _message_of(item: Any) -> Any:
    if isinstance(item, BaseMessage | str):
        return item
    if isinstance(item, Mapping) and item.get("type") == "messages" and "data" in item:  # typed stream parts
        return _message_of(item["data"])
    if not isinstance(item, tuple) or not item:
        return None
    if isinstance(item[0], BaseMessage):  # (message, metadata)
        return item[0]
    if isinstance(item[0], tuple):  # (namespace, payload) or (namespace, mode, payload) from subgraphs
        return _message_of(item[1] if len(item) == 2 else item[1:])
    if isinstance(item[0], str) and len(item) == 2:  # (mode, payload)
        return _message_of(item[1]) if item[0] == "messages" else None
    return None
