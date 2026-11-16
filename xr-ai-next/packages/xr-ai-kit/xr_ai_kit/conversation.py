# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Conversation``: multi-turn answers from one role, with tools, reading exactly the context it is given.

Per turn: the system prompt is ``instructions`` plus the listed context channels (rendered in order, each under its
name); the user message is the turn's text. The role runs a streaming tool loop (``role.as_llm()``): text is
spoken as it arrives on ``output``; tool calls run between rounds. A ``return_direct`` tool's result is the answer:
it is spoken as it arrives (a tool may return a stream of text) and the model is not asked again. An empty direct
result ends the turn with nothing more said (the tool spoke for itself, e.g. to everyone). A call that fails (invalid
arguments, an error before its first part) is not the answer: the model reads the error and may recover. In a round
that calls several tools, they run in order and the first direct one that succeeds ends the loop. The loop speaks
the answer; a tool returns its result (an app's own tool may also speak, as any app code can). It writes nothing
to any context itself (the voice component records the dialogue).
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from xr_ai_next import Component, trace
from xr_ai_next.clock import HeldStream
from xr_ai_next.context import ContextChannel, render
from xr_ai_next.llm import ChatMessage, ToolCall, ToolDef
from xr_ai_next.tools import ToolSpec, describe

if TYPE_CHECKING:
    from xr_ai_next import Participant, Role, Sink, Turn

__all__ = ["Conversation"]


class Conversation(Component):
    """Answers turns with ``role``, speaking on ``output``.

    ``context`` lists the channels the model reads (and nothing else). ``tools`` are ``@tool`` functions or
    ``xr_ai_tools.Tool`` objects. ``max_rounds`` bounds the tool loop; ``fallback`` is spoken when the loop fails
    before saying anything. ``name`` names it in the trace. It is a plain turn handler (:meth:`answer`); to route to it,
    wrap it: ``OneShotDestination(name, description, conversation.answer)`` (``xr_ai_kit.routing``).
    """

    def __init__(
        self,
        *,
        role: Role,
        output: Sink,
        instructions: str,
        context: Sequence[ContextChannel] = (),
        tools: Sequence[Any] = (),
        name: str = "conversation",
        max_rounds: int = 6,
        context_budget_bytes: int | None = None,
        max_tokens: int | None = None,
        fallback: str = "Sorry, I had trouble with that.",
    ) -> None:
        super().__init__()
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self._role = role
        self._output = output
        self._instructions = instructions.strip()
        self._context = list(context)
        self._tools: list[ToolSpec] = [describe(t) for t in tools]
        names = [t.name for t in self._tools]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names: {names}")
        self._name = name
        self._max_rounds = max_rounds
        self._budget = context_budget_bytes
        self._max_tokens = max_tokens
        self._fallback = fallback

    @property
    def name(self) -> str:
        return self._name

    async def answer(self, turn: Turn, p: Participant) -> None:
        """Answer ``turn`` for ``p``, speaking on the output."""
        messages = await self.messages(turn, p)
        await p.reply(self._run(messages, turn, p), to=self._output)

    async def messages(self, turn: Turn, p: Participant) -> list[ChatMessage]:
        """The transcript the model sees for this turn (system prompt with the context, then the turn)."""
        system = self._instructions
        if self._context and (text := await render(p, self._context, budget_bytes=self._budget)):
            system = f"{system}\n\n# Context\n\n{text}"
        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=turn.text)]

    def _definitions(self) -> list[ToolDef]:
        out = []
        for t in self._tools:
            schema = t.arguments.model_json_schema()
            out.append(ToolDef(name=t.name, description=t.description, parameters=schema))
        return out

    async def _run(self, messages: list[ChatMessage], turn: Turn, p: Participant) -> AsyncIterator[str]:
        llm = self._role.as_llm()
        definitions = self._definitions() or None
        transcript = list(messages)
        said = False
        for round_index in range(self._max_rounds):
            calls: list[ToolCall] = []
            text_parts: list[str] = []
            started_wall, started = time.time(), time.monotonic()
            events: HeldStream | None = None
            with trace.quiet_role_calls():  # this loop traces each round itself (below)
                try:
                    events = HeldStream(  # a real model call: a virtual-time loop waits for it for real
                        llm.stream_events(
                            transcript,
                            tools=definitions,
                            **({"max_tokens": self._max_tokens} if self._max_tokens else {}),
                        )
                    )
                    async for event in events:
                        if event.kind == "text" and event.text:
                            text_parts.append(event.text)
                            said = True
                            yield event.text
                        elif event.kind == "tool_call" and event.tool_call is not None:
                            calls.append(event.tool_call)
                except Exception as exc:  # noqa: BLE001 - a failed model call ends the turn politely
                    trace.emit("error", self._role.name, where="conversation", error=type(exc).__name__)
                    if not said:
                        yield self._fallback
                    return
                finally:
                    if events is not None:
                        await events.aclose()
            trace.emit(
                "role",
                self._role.name,
                method="chat",
                round=round_index,
                prompt=trace.brief(str(transcript[-1].content)),
                tool_calls=[c.name for c in calls],
                answer=trace.brief("".join(text_parts)),
                duration_s=time.monotonic() - started,
                t=started_wall,
            )
            if not calls:
                return
            transcript.append(ChatMessage(role="assistant", content="".join(text_parts), tool_calls=calls))
            for call in calls:
                result = await self._call_tool(call, turn, p)
                if isinstance(result, str):  # the model reads it
                    transcript.append(ChatMessage(role="tool", content=result, tool_call_id=call.id))
                    continue
                # A direct tool's answer: spoken as it arrives, and the turn ends with it (empty: nothing more to say).
                spoken = False
                try:
                    async with contextlib.aclosing(result):
                        async for part in result:
                            spoken = spoken or bool(part.strip())
                            yield part
                except Exception as exc:  # noqa: BLE001 - a stream that breaks off ends the turn politely
                    trace.emit("error", call.name, where="conversation", error=type(exc).__name__)
                    if not spoken and not said:
                        yield self._fallback
                return
        if not said:
            yield self._fallback

    async def _call_tool(self, call: ToolCall, turn: Turn, p: Participant) -> str | AsyncGenerator[str, None]:
        """The text the model reads for ``call``; for a ``return_direct`` tool that started, its answer instead."""
        spec = next((t for t in self._tools if t.name == call.name), None)
        if spec is None:
            return f"Unknown tool {call.name!r}."
        stream: AsyncGenerator[str, None] | None = None
        try:
            arguments = json.loads(call.arguments or "{}")
            if not spec.return_direct:
                return await spec.call(p, arguments, turn=turn)
            stream = spec.stream(p, arguments, turn=turn)
            first = await anext(stream, "")  # a failure before the first part goes to the model
        except Exception as exc:  # noqa: BLE001 - the model sees the failure and may recover
            if stream is not None:
                await stream.aclose()
            if isinstance(exc, ValidationError | json.JSONDecodeError):
                return f"Invalid arguments for {call.name}: {exc}"
            return f"The tool {call.name} failed: {type(exc).__name__}: {exc}"
        return _starting_with(first, stream)


async def _starting_with(first: str, rest: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    async with contextlib.aclosing(rest):
        yield first
        async for part in rest:
            yield part
