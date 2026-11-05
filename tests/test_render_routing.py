# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render conversation routing, shared transcripts, and participant lifetime."""

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from xr_ai_conversation import ContextManager
from xr_ai_interaction import Interrupted, ParticipantJoined, ParticipantLeft, ResponseOutput, UserQuery
from xr_ai_models import ChatStreamEvent, DecisionAnswer, DecisionResponse
from xr_ai_routing import AgentRouter
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.text_memory import RecallConversationRequest, TextMemoryTools
from xr_ai_voice import VOICE_OUTPUT_TOPIC
from xr_render_demo_eval import harness
from xr_render_demo_worker.agent import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    ConversationAgent,
    RenderAgent,
)
from xr_render_demo_worker.supervisor import SceneSupervisor


class _Decisions:
    def __init__(self, choices):
        self.choices = deque(choices)
        self.states = []

    async def decide(self, state, questions):
        self.states.append(state)
        choice = self.choices.popleft()
        return DecisionResponse(
            {"destination": DecisionAnswer(choice, 1.0, {
                name: float(name == choice) for name in questions["destination"].criteria
            })}, "fake", {}, {},
        )


class _LLM:
    def __init__(self, *answers, block=False):
        self.answers = deque(answers)
        self.messages = []
        self.block = block
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def stream_events(self, messages):
        self.messages.append(messages)
        answer = self.answers.popleft()
        try:
            yield ChatStreamEvent(kind="text", text=answer)
            self.started.set()
            if self.block:
                await asyncio.Event().wait()
        finally:
            self.cancelled.set()


class _Output(Agent):
    def __init__(self):
        super().__init__()
        self.items = []

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def receive(self, value: ResponseOutput, ctx: RuntimeContext):
        self.items.append((value, ctx.metadata))


@asynccontextmanager
async def _front_end(tmp_path, choices, llm, monkeypatch):
    memory = TextMemoryTools(tmp_path)
    fake = harness.FakeScene({}, harness._build_pose(), "", "", "")
    scene, tracking, *_ = fake.make_tools()
    supervisor = SceneSupervisor(llm=None, scene=scene, tracking=tracking, text_memory=memory, subagent_tools=[])
    scene_prompts = []

    async def tool_loop(messages, tools, call_model, **kwargs):
        scene_prompts.append(messages[-1].content)
        return SimpleNamespace(content="Which object?", messages=list(messages), tool_calls=())

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", tool_loop)
    runtime = AgentRuntime()
    context = ContextManager(llm)
    conversation = ConversationAgent(llm, context=context, runtime=runtime, text_memory=memory)
    output = runtime.register("output", _Output())
    decisions = _Decisions(choices)
    departed = []
    idle = []
    changed = asyncio.Event()

    async def cleanup(participant):
        supervisor.forget_participant(participant)
        await context.clear(participant)
        departed.append(participant)

    async def status(value, participant):
        if value == "idle":
            idle.append(participant)
            changed.set()

    async def wait_idle(count):
        while len(idle) < count:
            changed.clear()
            await asyncio.wait_for(changed.wait(), 2)

    async def send(text, participant="alice"):
        count = len(idle)
        await runtime.publish(USER_QUERY_TOPIC, UserQuery(text=text, timestamp_us=time.time_ns() // 1_000),
                              participant_id=participant, source="test")
        await wait_idle(count + 1)

    router = AgentRouter(
        {"conversation": conversation, "scene": RenderAgent(supervisor, runtime)}, decisions,
        query_topic=USER_QUERY_TOPIC, output_topic=VOICE_OUTPUT_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC, participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC, cleanup_participant=cleanup, set_status=status,
    )
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        try:
            yield SimpleNamespace(runtime=runtime, router=router, send=send, wait_idle=wait_idle,
                                  output=output, context=context, scene_prompts=scene_prompts,
                                  memory=memory, decisions=decisions, departed=departed)
        finally:
            await router.stop()
    await conversation.close()
    await context.close()


async def test_general_question_streams_without_calling_supervisor(tmp_path, monkeypatch):
    llm = _LLM("They migrate to find food.")
    async with _front_end(tmp_path, ["conversation"], llm, monkeypatch) as app:
        await app.send("Why do swallows migrate?")
        assert app.scene_prompts == []
        chunks = [item for item, _ in app.output.items]
        assert [item.text for item in chunks] == ["They migrate to find food.", ""]
        assert [item.final for item in chunks] == [False, True]
        assert len({item.request_id for item in chunks}) == 1
        assert {meta.source for _, meta in app.output.items} == {"conversation"}
        recalled = await app.memory.recall_conversation.execute(RecallConversationRequest(participant_id="alice"))
        assert [entry.text for entry in recalled.entries] == ["Why do swallows migrate?", "They migrate to find food."]


async def test_scene_and_conversation_keep_history_in_both_directions(tmp_path, monkeypatch):
    llm = _LLM("Beryl is a mineral.", "We were discussing beryl.")
    async with _front_end(tmp_path, ["conversation", "scene", "conversation"], llm, monkeypatch) as app:
        await app.send("What is beryl?")
        await app.send("Make that mineral's color the sphere's color.")
        assert "Beryl is a mineral." in app.scene_prompts[0]
        await app.send("Explain the mineral further.")
        context = llm.messages[-1][1].content
        assert "What is beryl?" in context
        assert "Which object?" in context
        assert "Make that mineral's color" in context
        assert [item["user"] for item in app.decisions.states[-1]["completed_history"]] == [
            "What is beryl?", "Make that mineral's color the sphere's color.",
        ]
        recalled = await app.memory.recall_conversation.execute(RecallConversationRequest(participant_id="alice"))
        assert len(recalled.entries) == 6


async def test_compound_scene_request_reaches_supervisor_whole(tmp_path, monkeypatch):
    async with _front_end(tmp_path, ["scene"], _LLM(), monkeypatch) as app:
        query = "Shift the box behind me, make it amber, and explain amber."
        await app.send(query)
        assert len(app.scene_prompts) == 2
        assert "Verified scene changes this turn: none" in app.scene_prompts[1]
        assert f"User request: {query}" in app.scene_prompts[0]


@pytest.mark.parametrize("event, topic", [
    (Interrupted(), INTERRUPTED_TOPIC), (ParticipantLeft(), PARTICIPANT_LEFT_TOPIC),
])
async def test_interrupted_conversation_closes_stream_without_recording(tmp_path, monkeypatch, event, topic):
    llm = _LLM("A partial answer.", block=True)
    async with _front_end(tmp_path, ["conversation"], llm, monkeypatch) as app:
        await app.runtime.publish(USER_QUERY_TOPIC, UserQuery(text="Explain migration.", timestamp_us=1),
                                  participant_id="alice", source="test")
        await asyncio.wait_for(llm.started.wait(), 2)
        await app.runtime.publish(topic, event, participant_id="alice", source="voice")
        await app.wait_idle(1)
        assert llm.cancelled.is_set()
        assert [(item.text, item.final) for item, _ in app.output.items] == [
            ("A partial answer.", False), ("", True),
        ]
        recalled = await app.memory.recall_conversation.execute(RecallConversationRequest(participant_id="alice"))
        assert recalled.entries == []
        assert app.context.blocks("alice") == ()


async def test_rejoin_clears_working_context_but_keeps_persistent_transcripts(tmp_path, monkeypatch):
    llm = _LLM("You prefer amber.", "Hello again.")
    async with _front_end(tmp_path, ["conversation", "conversation"], llm, monkeypatch) as app:
        await app.send("My preferred color is amber.")
        await app.runtime.publish(PARTICIPANT_LEFT_TOPIC, ParticipantLeft(), participant_id="alice", source="voice")
        assert app.context.blocks("alice") == ()
        await app.runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="alice", source="voice")
        await app.send("Hello.")
        assert "amber" not in llm.messages[-1][1].content
        assert app.decisions.states[-1]["completed_history"] == []
        recalled = await app.memory.recall_conversation.execute(RecallConversationRequest(participant_id="alice"))
        assert len(recalled.entries) == 4
        assert app.departed == ["alice"]


async def test_participants_do_not_share_conversation_context(tmp_path, monkeypatch):
    llm = _LLM("Your code is zircon.", "I don't know your code.")
    async with _front_end(tmp_path, ["conversation", "conversation"], llm, monkeypatch) as app:
        await app.send("Remember my code: zircon.", "alice")
        await app.send("What is my code?", "bob")
        assert "zircon" not in llm.messages[-1][1].content
        assert app.decisions.states[-1]["completed_history"] == []
        assert {metadata.participant_id for _, metadata in app.output.items} == {"alice", "bob"}


async def test_worker_drains_router_before_closing_models(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, Mock

    from xr_render_demo_worker import app

    llm = _LLM("An unfinished answer.", block=True)

    async def close_llm():
        assert llm.cancelled.is_set()

    llm.close = AsyncMock(side_effect=close_llm)
    decisions = _Decisions(["conversation"])
    decisions.close = AsyncMock()
    vlm = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(app, "make_llm", lambda *_: llm)
    monkeypatch.setattr(app, "make_decision", lambda *_: decisions)
    monkeypatch.setattr(app, "make_vlm", lambda *_: vlm)
    monkeypatch.setattr(app, "make_tts", Mock())
    monkeypatch.setattr(app, "setup_logging", Mock())
    monkeypatch.setattr(app, "load_models_config", Mock())
    monkeypatch.setattr(app, "load_voice_gate_config", Mock())
    for name in ("SceneTools", "TrackingTools", "CurrentFrameTool", "HubVoiceTransport", "SceneSupervisor",
                 "XRSessionController"):
        monkeypatch.setattr(app, name, Mock())
    monkeypatch.setattr(app, "close_clients", AsyncMock())
    options = {}

    class Voice(Agent):
        async def run(self, runtime):
            await runtime.publish(USER_QUERY_TOPIC, UserQuery(text="Tell me about migration.", timestamp_us=1),
                                  participant_id="alice", source="voice")
            await asyncio.wait_for(llm.started.wait(), 2)

    def make_voice(**kwargs):
        options.update(kwargs)
        return Voice()

    monkeypatch.setattr(app, "VoiceAgent", SimpleNamespace(from_models=make_voice))
    config = SimpleNamespace(
        models_config=tmp_path / "models.json", voice_gate_yaml=tmp_path / "gate.yaml",
        silence_duration=0.5, min_speech=0.2, silero_threshold=0.5, idle_timeout_secs=None,
        scene_endpoint="unused", openxr_endpoint="unused", text_memory_dir=tmp_path, video_history_enabled=False,
    )
    await app.run_app(config)
    llm.close.assert_awaited_once()
    decisions.close.assert_awaited_once()
    vlm.close.assert_awaited_once()
    assert options["participant_joined_topic"] == PARTICIPANT_JOINED_TOPIC
    assert options["participant_left_topic"] == PARTICIPANT_LEFT_TOPIC
    app.SceneSupervisor.return_value.handle.assert_not_called()
    app.CurrentFrameTool.return_value.release.assert_called_once_with("alice")


async def test_scene_query_supersedes_streamed_conversation(tmp_path, monkeypatch):
    llm = _LLM("An unfinished explanation.", block=True)
    async with _front_end(tmp_path, ["conversation", "scene"], llm, monkeypatch) as app:
        await app.runtime.publish(USER_QUERY_TOPIC, UserQuery(text="Explain bird migration.", timestamp_us=1),
                                  participant_id="alice", source="voice")
        await asyncio.wait_for(llm.started.wait(), 2)
        await app.send("What virtual objects are here?")
        await app.wait_idle(2)
        assert llm.cancelled.is_set()
        assert len(app.scene_prompts) == 1
        assert "An unfinished explanation." not in app.scene_prompts[0]
        recalled = await app.memory.recall_conversation.execute(RecallConversationRequest(participant_id="alice"))
        assert [entry.text for entry in recalled.entries] == ["What virtual objects are here?", "Which object?"]


async def test_routing_eval_reports_a_wrong_destination(capsys):
    from xr_render_demo_eval.routing import CASES, run_case

    assert not await run_case(CASES[0], _Decisions(["scene"]))
    assert "FAIL general_knowledge" in capsys.readouterr().out
