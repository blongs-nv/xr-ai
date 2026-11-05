# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score the live decision model against the render front end's route descriptions."""

import argparse
import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

from xr_ai_conversation import ContextManager
from xr_ai_models import ChatStreamEvent, load_models_config, make_decision
from xr_ai_routing import AgentExchange, AgentRouter, AgentTurn
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.text_memory import TextMemoryTools
from xr_render_demo_worker.agent import ConversationAgent, RenderAgent
from xr_render_demo_worker.models import SceneReply


@dataclass(frozen=True)
class RoutingCase:
    name: str
    query: str
    expected: str
    history: tuple[AgentExchange, ...] = ()


CASES = (
    RoutingCase("general_knowledge", "Why do migrating birds fly in a V?", "conversation"),
    RoutingCase("greeting", "Good afternoon!", "conversation"),
    RoutingCase("general_follow_up", "What about at night?", "conversation", (
        AgentExchange("How do migrating birds navigate?", "They use landmarks, the sun, and stars."),
    )),
    RoutingCase("scene_create", "Put an amber ball on the floor.", "scene"),
    RoutingCase("scene_inspection", "How many virtual objects are there?", "scene"),
    RoutingCase("scene_follow_up", "Make that smaller.", "scene", (
        AgentExchange("Add a box beside me.", "I added a box beside you."),
    )),
    RoutingCase("scene_after_conversation", "Paint the box that color.", "scene", (
        AgentExchange("What color is lapis lazuli?", "It is a deep blue."),
    )),
    RoutingCase("conversation_after_scene", "Why does amber form?", "conversation", (
        AgentExchange("Make the box amber.", "I made it amber."),
    )),
    RoutingCase("compound_scene", "Move the ball behind me and make the box violet.", "scene"),
    RoutingCase("mixed_scene_and_general", "Create a ball and explain how gravity works.", "scene"),
    RoutingCase("mixed_general_first", "Explain evaporation and make the ball pale green.", "scene"),
    RoutingCase("camera_present", "What am I holding in my hand?", "scene"),
    RoutingCase("camera_past", "What was on the desk a minute ago?", "scene"),
    RoutingCase("exact_recall", "What shape did I ask you to create first?", "scene"),
    RoutingCase("incomplete_action", "Move the box to the", "scene"),
)


class _ConversationModel:
    async def stream_events(self, _messages):
        yield ChatStreamEvent(kind="text", text="conversation")


class _Supervisor:
    async def handle(self, _request):
        return SceneReply(response="scene")


async def _control(_turn, _source, _text):
    pass


async def run_case(case: RoutingCase, decisions) -> bool:
    runtime = AgentRuntime()
    context = ContextManager(_ConversationModel())
    with tempfile.TemporaryDirectory(prefix="xr-render-routing-") as directory:
        conversation = ConversationAgent(
            _ConversationModel(), context=context, runtime=runtime, text_memory=TextMemoryTools(directory),
        )
        router = AgentRouter(
            {"conversation": conversation, "scene": RenderAgent(_Supervisor(), runtime)},
            decisions, output=_control,
        )
        async with runtime:
            try:
                result = await router.handle(AgentTurn("eval-user", case.name, case.query, history=case.history))
            finally:
                await router.stop()
                await conversation.close()
                await context.close()
    passed = result.disposition == "answer" and result.text == case.expected
    print(f"{'PASS' if passed else 'FAIL'} {case.name}: expected={case.expected} actual={result.text!r}")
    return passed


async def _run(cases, config):
    decisions = make_decision(load_models_config(config), "routing")
    try:
        results = [await run_case(case, decisions) for case in cases]
    finally:
        await decisions.close()
    print(f"{sum(results)}/{len(results)} routing cases passed")
    return all(results)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case names; defaults to every case")
    parser.add_argument("--models-config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "yaml/models.json")
    args = parser.parse_args()
    unknown = set(args.cases) - {case.name for case in CASES}
    if unknown:
        parser.error(f"unknown cases: {', '.join(sorted(unknown))}")
    cases = [case for case in CASES if not args.cases or case.name in args.cases]
    raise SystemExit(0 if asyncio.run(_run(cases, args.models_config)) else 1)
