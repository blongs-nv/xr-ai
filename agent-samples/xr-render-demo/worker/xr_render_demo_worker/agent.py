# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routed scene and conversation destinations for the render worker."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from loguru import logger
from xr_ai_conversation import ContextManager, Conversation
from xr_ai_interaction import Interrupted, ParticipantJoined, ParticipantLeft, ResponseOutputs, UserQuery
from xr_ai_models import LLMService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutingAid
from xr_ai_runtime import AgentRuntime, Topic
from xr_ai_tools.text_memory import TextMemoryTools
from xr_ai_voice import VOICE_OUTPUT_TOPIC

from .models import SceneRequest
from .transcript import persist_exchange

USER_QUERY_TOPIC = Topic("xr-render.user-query", UserQuery)
PARTICIPANT_JOINED_TOPIC = Topic("xr-render.participant-joined", ParticipantJoined)
PARTICIPANT_LEFT_TOPIC = Topic("xr-render.participant-left", ParticipantLeft)
INTERRUPTED_TOPIC = Topic("xr-render.interrupted", Interrupted)

_SCENE_ROUTING = RoutingAid(
    "scene",
    "Visual questions about the user's surroundings, objects they hold, or what the camera sees, now or earlier. "
    "Commands to create, inspect, move, recolor, resize, remove, or undo edits to virtual 3D objects and shapes. "
    "Exact conversation transcript recall. Requests combining a scene command with a general question belong "
    "here in full: perform the scene action and answer the question together.",
)
_CONVERSATION_ROUTING = RoutingAid(
    "conversation",
    "General knowledge questions, explanations, greetings, and casual chat. Eligible only when the entire "
    "request requires no scene action, object inspection, camera observation, or exact transcript recall.",
)


class RenderAgent:
    """Delegate a complete scene turn to the supervisor and publish its verified reply."""

    def __init__(self, supervisor, runtime: AgentRuntime) -> None:
        self._supervisor = supervisor
        self._runtime = runtime

    def routing_aid(self, _participant_id: str) -> RoutingAid:
        return _SCENE_ROUTING

    async def admit(self, _turn: AgentTurn, *, focused: bool) -> Admission:
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        await started()
        try:
            reply = await self._supervisor.handle(
                SceneRequest(
                    transcript=turn.query,
                    participant_id=turn.participant_id,
                    timestamp_us=turn.timestamp_us or 0,
                    trace_id=turn.turn_id,
                )
            )
            text = reply.response
        except Exception:  # noqa: BLE001
            # isolation boundary: after scene work starts, this destination owns its failure reply.
            logger.exception("xr-render turn failed for {}", turn.participant_id)
            text = "Something went wrong. Please try again."
        async with ResponseOutputs(self._runtime, VOICE_OUTPUT_TOPIC, source="scene").reply(
            turn.participant_id,
            request_id=turn.turn_id,
            timestamp_us=turn.timestamp_us,
        ) as answer:
            await answer.send(text)
        return DestinationResult(text=text)

    async def release(self, _participant_id: str) -> None:
        pass


class ConversationAgent(Conversation):
    """Persist completed general conversation for the supervisor's transcript recall."""

    def __init__(
        self,
        llm: LLMService,
        *,
        context: ContextManager,
        runtime: AgentRuntime,
        text_memory: TextMemoryTools,
    ) -> None:
        super().__init__(
            llm,
            context=context,
            instructions=Path(__file__).with_name("conversation_prompt.txt").read_text(encoding="utf-8"),
            routing=_CONVERSATION_ROUTING,
            runtime=runtime,
            output_topic=VOICE_OUTPUT_TOPIC,
        )
        self._text_memory = text_memory

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        result = await super().answer(turn, started=started)
        await persist_exchange(
            self._text_memory,
            participant_id=turn.participant_id,
            query=turn.query,
            response=result.text,
            timestamp_us=turn.timestamp_us or 0,
        )
        return result


__all__ = [
    "ConversationAgent",
    "INTERRUPTED_TOPIC",
    "PARTICIPANT_JOINED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "USER_QUERY_TOPIC",
    "RenderAgent",
]
