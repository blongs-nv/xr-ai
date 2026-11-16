# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in research agent that speaks A2A (the Agent2Agent protocol).

Any A2A agent can sit behind sample 07. This one fakes a slow research job so
the sample runs on its own: three progress updates, then a result. It is
written with the official A2A Python SDK (`a2a-sdk`) and served by uvicorn;
it does not use XR-AI at all.

`uv run delegate_a2a` (the sample on DeviceIOHub) starts it with the hub
and the app. For a dry run, start it first, from this directory (the front
end expects it on port 8830):

  uv --config-file ../../../../uv.toml run research_agent

What the front end sees, in A2A terms: a task that is WORKING with a status
message per step, one text artifact, then COMPLETED. Cancelling the task
(`tasks/cancel`) ends it as CANCELED.
"""

import argparse
import asyncio
from pathlib import Path

import uvicorn
from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from starlette.applications import Starlette

STEPS = ("searching maps", "reading reviews", "comparing")


class ResearchAgent(AgentExecutor):
    """Reports each step as the task's status, then one text artifact."""

    def __init__(self, step_s: float = 5.0) -> None:
        self.step_s = step_s

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )
        request = get_message_text(context.message)
        for step in STEPS:
            await updater.start_work(new_text_message(step))
            await asyncio.sleep(self.step_s)
        await updater.add_artifact(
            parts=[new_text_part(text=f"Three options for: {request} ...")],
            name="result",
        )
        await updater.complete()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.cancel(new_text_message("cancelled"))


def build_app(
    host: str = "127.0.0.1", port: int = 8830, step_s: float = 5.0
) -> Starlette:
    """The agent as an ASGI app; its card advertises `host:port`."""
    card = AgentCard(
        name="Research agent",
        description="Finds and compares options; takes a while.",
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=f"http://{host}:{port}",
                protocol_version="1.0",
            )
        ],
        skills=[
            AgentSkill(
                id="research",
                name="Research",
                description="Search, read and compare; report options.",
                tags=["research"],
                examples=["find me a quiet cafe near the station"],
            )
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=ResearchAgent(step_s=step_s),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, "/"),
    ]
    return Starlette(routes=routes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8830)
    parser.add_argument("--step-s", type=float, default=5.0)
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="created once the agent listens (the samples' launcher)",
    )
    args = parser.parse_args()
    app = build_app(args.host, args.port, args.step_s)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port))
    asyncio.run(serve(server, args.ready_file))


async def serve(server: uvicorn.Server, ready_file: Path | None) -> None:
    """Run `server`; create `ready_file` once it accepts connections."""
    serving = asyncio.ensure_future(server.serve())
    if ready_file is not None:
        while not server.started and not serving.done():
            await asyncio.sleep(0.05)
        if server.started:
            ready_file.touch()
    await serving


if __name__ == "__main__":
    main()
