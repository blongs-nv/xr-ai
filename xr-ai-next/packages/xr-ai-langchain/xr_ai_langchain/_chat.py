# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``chat_model(role)``: a role as a LangChain chat model, over ``role.as_llm()``.

The profile still decides which model fills the role. Calls go through the role's ``LLMService`` (tools, streaming,
the role's concurrency limit), are cancelled with the calling task, and each emits one ``role`` trace event
(``method="chat"``) as ``Conversation`` does.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import ChatMessage as LCChatMessage
from langchain_core.messages.tool import invalid_tool_call, tool_call, tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr
from xr_ai_next import clock, trace
from xr_ai_next.llm import ChatMessage, ChatResponse, ImagePart, TextPart, ToolCall, ToolDef

__all__ = ["RoleChatModel", "chat_model", "to_chat_messages", "to_langchain_messages"]

_ROLES = {"system": "system", "user": "user", "human": "user", "assistant": "assistant", "ai": "assistant"}


def chat_model(
    role: Any,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
) -> RoleChatModel:
    """``role`` as a LangChain ``BaseChatModel`` (async: ``ainvoke``, ``astream``, ``bind_tools``).

    ``max_tokens``, ``temperature`` and ``enable_thinking`` override the profile entry for every call; ``None``
    keeps the entry's own. Building it calls no model, so a module-level ``chat_model(app.role(...))`` is fine.
    """
    return RoleChatModel(role=role, max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking)


class RoleChatModel(BaseChatModel):
    """A LangChain chat model over a role (build it with :func:`chat_model`).

    - **Async only.** The role's clients and limits belong to the app's event loop, so ``invoke`` and ``stream``
      raise ``NotImplementedError``; use ``ainvoke``, ``astream`` or a LangGraph graph run with ``ainvoke`` or
      ``astream``.
    - **Tools.** ``bind_tools`` accepts anything LangChain converts to an OpenAI function tool. Requested calls come
      back as ``AIMessage.tool_calls`` (``invalid_tool_calls`` when the arguments are not a JSON object).
      ``tool_choice`` is accepted and not enforced: a role's ``LLMService`` lets the model choose.
    - **Messages.** System, human, AI (with tool calls) and tool messages map to ``xr_ai_models.ChatMessage``;
      consecutive system messages are merged into one, since many chat templates accept only one. Text and image
      content blocks are kept; other blocks (reasoning, provider-specific) are dropped.
    - **Reasoning** never enters ``content``; it is in ``additional_kwargs["reasoning_content"]``.
    - ``stop`` sequences are not supported and raise ``ValueError``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: Any = Field(exclude=True)
    """The role (``app.role(name)``)."""
    max_tokens: int | None = None
    temperature: float | None = None
    enable_thinking: bool | None = None

    _service: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "xr-ai-role"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "role": self._role_name(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
        }

    def _role_name(self) -> str:
        return str(getattr(self.role, "name", "role"))

    def _llm(self) -> Any:
        # One service per chat model: the role resolves lazily (a test session turns it into a fake in place), and
        # one instance keeps a fake role's tool-call ids distinct across rounds.
        if self._service is None:
            self._service = self.role.as_llm()
        return self._service

    # ------------------------------------------------------------------ tools

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """This model with ``tools`` offered on every call (OpenAI function-tool format in the bound kwargs)."""
        del tool_choice  # not enforceable through an LLMService; documented on the class
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    # ------------------------------------------------------------------ calls

    def _options(self, stop: list[str] | None, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        if stop:
            raise ValueError("xr-ai roles do not support stop sequences")
        options: dict[str, Any] = {}
        tools = [_tool_def(t) for t in kwargs.get("tools") or ()]
        if tools:
            options["tools"] = tools
        for name in ("max_tokens", "temperature", "enable_thinking"):
            value = kwargs.get(name, getattr(self, name))
            if value is not None:
                options[name] = value
        return options

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("chat_model(role) is async: use ainvoke/astream (roles run on the app's loop)")

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError("chat_model(role) is async: use ainvoke/astream (roles run on the app's loop)")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        options = self._options(stop, kwargs)
        transcript = to_chat_messages(messages)
        started_wall, started = time.time(), time.monotonic()
        # A real model call (held: a virtual clock waits for it); its own role event would duplicate this call's.
        try:
            with trace.quiet_role_calls(), clock.held():
                response: ChatResponse = await self._llm().chat(transcript, **options)
        except Exception as exc:
            trace.emit("error", self._role_name(), where="chat_model", error=type(exc).__name__)
            raise
        message = _ai_message(response)
        self._trace(message, started_wall, started)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # The base class reports each chunk to ``run_manager`` (LangGraph's message stream); doing it here too would
        # stream every token twice.
        options = self._options(stop, kwargs)
        transcript = to_chat_messages(messages)
        started_wall, started = time.time(), time.monotonic()
        text: list[str] = []
        calls: list[ToolCall] = []
        events = clock.HeldStream(self._llm().stream_events(transcript, **options))  # held until it ends or closes
        async with contextlib.aclosing(events):
            while True:
                # Quiet around each step, not across ``yield``: the consumer may resume this generator elsewhere.
                try:
                    with trace.quiet_role_calls():
                        event = await anext(events)
                except StopAsyncIteration:
                    break
                except Exception as exc:
                    trace.emit("error", self._role_name(), where="chat_model", error=type(exc).__name__)
                    raise
                chunk = _chunk(event, len(calls))
                if event.kind == "text" and event.text:
                    text.append(event.text)
                elif event.kind == "tool_call" and event.tool_call is not None:
                    calls.append(event.tool_call)
                if chunk is not None:
                    yield chunk
        summary = AIMessage(content="".join(text), tool_calls=[_tool_call(c) for c in calls if _args(c) is not None])
        self._trace(summary, started_wall, started)

    def _trace(self, message: AIMessage, started_wall: float, started: float) -> None:
        trace.emit(
            "role",
            self._role_name(),
            method="chat",
            via="langchain",
            tool_calls=[c["name"] for c in message.tool_calls],
            answer=trace.brief(_text(message.content)),
            duration_s=time.monotonic() - started,
            t=started_wall,
        )


# ---------------------------------------------------------------------- conversion


def _tool_def(tool: Any) -> ToolDef:
    spec = tool if isinstance(tool, ToolDef) else convert_to_openai_tool(tool)
    if isinstance(spec, ToolDef):
        return spec
    function = spec.get("function", spec)
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    return ToolDef(name=function["name"], description=function.get("description", ""), parameters=parameters)


def _args(call: ToolCall) -> dict[str, Any] | None:
    try:
        value = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _tool_call(call: ToolCall) -> Any:
    return tool_call(name=call.name, args=_args(call) or {}, id=call.id)


def _ai_message(response: ChatResponse) -> AIMessage:
    calls, invalid = [], []
    for call in response.tool_calls or ():
        args = _args(call)
        if args is None:
            error = "arguments are not a JSON object"
            invalid.append(invalid_tool_call(name=call.name, args=call.arguments, id=call.id, error=error))
        else:
            calls.append(tool_call(name=call.name, args=args, id=call.id))
    extra = {"reasoning_content": response.reasoning} if response.reasoning else {}
    return AIMessage(
        content=response.content or "",
        tool_calls=calls,
        invalid_tool_calls=invalid,
        additional_kwargs=extra,
        response_metadata={"finish_reason": response.finish_reason} if response.finish_reason else {},
    )


def _chunk(event: Any, index: int) -> ChatGenerationChunk | None:
    if event.kind == "text":
        return ChatGenerationChunk(message=AIMessageChunk(content=event.text)) if event.text else None
    if event.kind == "reasoning":
        if not event.reasoning:
            return None
        reasoning = {"reasoning_content": event.reasoning}
        return ChatGenerationChunk(message=AIMessageChunk(content="", additional_kwargs=reasoning))
    if event.kind == "tool_call" and event.tool_call is not None:
        call = event.tool_call
        part = tool_call_chunk(name=call.name, args=call.arguments or "{}", id=call.id, index=index)
        return ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[part]))
    if event.kind == "finish" and event.finish_reason:
        return ChatGenerationChunk(
            message=AIMessageChunk(content=""), generation_info={"finish_reason": event.finish_reason}
        )
    return None


def _text(content: Any) -> str:
    """The text of LangChain message content: a string, or the text blocks of a block list."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or ():
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _image_url(block: Mapping[str, Any]) -> str | None:
    if block.get("type") == "image_url":
        value = block.get("image_url")
        return value.get("url") if isinstance(value, Mapping) else value
    if block.get("type") == "image":
        if block.get("url"):
            return str(block["url"])
        data = block.get("base64") or block.get("data")
        if data:
            return f"data:{block.get('mime_type', 'image/jpeg')};base64,{data}"
    return None


def _content(content: Any) -> str | list[Any]:
    if isinstance(content, str):
        return content
    parts: list[Any] = []
    for block in content or ():
        if isinstance(block, str):
            parts.append(TextPart(block))
        elif isinstance(block, Mapping):
            if block.get("type") == "text":
                parts.append(TextPart(str(block.get("text", ""))))
            elif (url := _image_url(block)) is not None:
                parts.append(ImagePart(url))
    if all(isinstance(p, TextPart) for p in parts):
        return "".join(p.text for p in parts)
    return parts


def to_chat_messages(messages: Sequence[BaseMessage]) -> list[ChatMessage]:
    """LangChain messages as ``xr_ai_models.ChatMessage`` (consecutive system messages merged into one)."""
    out: list[ChatMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            out.append(ChatMessage(role="tool", content=_text(message.content), tool_call_id=message.tool_call_id))
            continue
        elif isinstance(message, LCChatMessage) and message.role in _ROLES:
            role = _ROLES[message.role]
        else:
            raise TypeError(f"cannot send a {type(message).__name__} to an xr-ai role")
        if role == "assistant":
            calls = [
                ToolCall(id=c.get("id") or f"call_{i}", name=c["name"], arguments=json.dumps(c.get("args") or {}))
                for i, c in enumerate(getattr(message, "tool_calls", None) or ())
            ]
            out.append(ChatMessage(role="assistant", content=_text(message.content), tool_calls=calls or None))
            continue
        content = _content(message.content)
        previous = out[-1] if out else None
        if role == "system" and previous is not None and previous.role == "system":
            out[-1] = ChatMessage(role="system", content=f"{_text_of(previous.content)}\n\n{_text_of(content)}")
            continue
        out.append(ChatMessage(role=role, content=content))
    return out


def _text_of(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.text for p in content if isinstance(p, TextPart))


def to_langchain_messages(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
    """``xr_ai_models.ChatMessage`` as LangChain messages (text and image parts as content blocks)."""
    out: list[BaseMessage] = []
    for message in messages:
        content: Any = message.content
        if not isinstance(content, str):
            content = [
                {"type": "text", "text": p.text}
                if isinstance(p, TextPart)
                else {"type": "image_url", "image_url": {"url": p.url}}
                for p in content
                if isinstance(p, TextPart | ImagePart)
            ]
        if message.role == "system":
            out.append(SystemMessage(content=content))
        elif message.role == "user":
            out.append(HumanMessage(content=content))
        elif message.role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=message.tool_call_id or ""))
        else:
            calls = message.tool_calls or ()
            out.append(AIMessage(content=content, tool_calls=[_tool_call(c) for c in calls]))
    return out
