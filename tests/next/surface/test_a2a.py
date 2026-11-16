# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A2ABackend against real A2A servers (the official SDK, served by uvicorn on a free 127.0.0.1 port).

The happy path and cancel use sample 07's own research agent; failure and input-required use small agents here.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

pytest.importorskip("a2a.server.routes")
import uvicorn  # noqa: E402
from a2a.client import ClientConfig, create_client  # noqa: E402
from a2a.helpers import new_task_from_user_message, new_text_message  # noqa: E402
from a2a.server.agent_execution import AgentExecutor, RequestContext  # noqa: E402
from a2a.server.events import EventQueue  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater  # noqa: E402
from a2a.types import (  # noqa: E402
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    ListTasksRequest,
    TaskState,
)
from starlette.applications import Starlette  # noqa: E402
from xr_ai_kit.delegate import A2ABackend, Job  # noqa: E402
from xr_ai_next.testing import FakeModels, ToolCall  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[3] / "xr-ai-next" / "samples" / "sdk-tour" / "07-delegate-a2a"


def _research_agent_module():
    spec = importlib.util.spec_from_file_location("sample07_research_agent", SAMPLE / "research_agent" / "agent.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.asynccontextmanager
async def _serve(make_app) -> AsyncIterator[str]:  # noqa: ANN001 - make_app(port) -> ASGI app
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(make_app(port), host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.02)
    assert server.started, "the A2A test server did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


def _agent_app(executor: AgentExecutor, port: int) -> Starlette:
    card = AgentCard(
        name="test agent",
        description="test",
        version="0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=f"http://127.0.0.1:{port}", protocol_version="1.0")
        ],
        skills=[AgentSkill(id="test", name="test", description="test", tags=["test"])],
    )
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])


class _Ending(AgentExecutor):
    """Works for a moment, then ends the task in ``state`` with ``text``."""

    def __init__(self, state: str, text: str) -> None:
        self.state, self.text = state, text

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        await updater.start_work(new_text_message("starting"))
        await getattr(updater, self.state)(new_text_message(self.text))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue=event_queue, task_id=context.task_id, context_id=context.context_id)
        await updater.cancel()


async def _task_states(url: str) -> list[int]:
    client = await create_client(url, ClientConfig(streaming=False))
    try:
        listed = await client.list_tasks(ListTasksRequest())
        return [task.status.state for task in listed.tasks]
    finally:
        await client.close()


def _job(request: str = "find a quiet cafe") -> Job:
    return Job(id="j1", request=request, participant="p1")


async def test_progress_then_result_from_the_sample_agent() -> None:
    agent = _research_agent_module()
    async with _serve(lambda port: agent.build_app(port=port, step_s=0.01)) as url:
        updates = [u async for u in A2ABackend(url).run("find a quiet cafe", _job())]
    assert [(u.kind, u.text) for u in updates] == [
        ("progress", "searching maps"),
        ("progress", "reading reviews"),
        ("progress", "comparing"),
        ("result", "Three options for: find a quiet cafe ..."),
    ]


async def test_cancelling_the_job_cancels_the_task_on_the_agent() -> None:
    agent = _research_agent_module()
    async with _serve(lambda port: agent.build_app(port=port, step_s=30)) as url:
        seen: list[str] = []

        async def consume() -> None:
            async for update in A2ABackend(url).run("plan a trip", _job("plan a trip")):
                seen.append(update.text)

        task = asyncio.create_task(consume())
        for _ in range(200):
            if seen:
                break
            await asyncio.sleep(0.02)
        assert seen == ["searching maps"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await _task_states(url) == [TaskState.TASK_STATE_CANCELED]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("failed", "the agent failed: no maps today"),
        ("reject", "the agent rejected the request: no maps today"),
    ],
)
async def test_a_failed_task_is_an_error(state: str, expected: str) -> None:
    async with _serve(lambda port: _agent_app(_Ending(state, "no maps today"), port)) as url:
        updates = [u async for u in A2ABackend(url).run("x", _job("x"))]
        assert [(u.kind, u.text) for u in updates] == [("progress", "starting"), ("error", expected)]
        assert await _task_states(url) in ([TaskState.TASK_STATE_FAILED], [TaskState.TASK_STATE_REJECTED])


async def test_input_required_is_an_error_and_the_task_is_cancelled() -> None:
    """A delegated job cannot answer the agent: the job fails with the agent's question, and the waiting task
    is cancelled on the agent rather than left open."""
    async with _serve(lambda port: _agent_app(_Ending("requires_input", "which city?"), port)) as url:
        updates = [u async for u in A2ABackend(url).run("x", _job("x"))]
        assert updates[-1].kind == "error"
        assert updates[-1].text == "the agent asked for input, which a delegated job cannot give: which city?"
        assert await _task_states(url) == [TaskState.TASK_STATE_CANCELED]


async def test_without_the_sdk_the_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def no_a2a(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if name == "a2a" or name.startswith("a2a."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_a2a)
    with pytest.raises(RuntimeError, match=r"xr-ai-kit\[a2a\]"):
        async for _ in A2ABackend("http://127.0.0.1:9").run("x", _job("x")):
            pass


async def test_sample_07_end_to_end_with_its_research_agent(load_sample) -> None:
    """The front end delegates to the real research agent; progress lands in context, the result is spoken."""
    m = load_sample("07-delegate-a2a")
    agent = _research_agent_module()
    models = FakeModels(
        {
            "answer_with_reasoning": [
                ToolCall("start_research", {"request": "quiet cafe near the station"}),
                "I'm on it.",
            ],
            "answer_lightweight": ["I found three options near the station."],
        }
    )
    async with _serve(lambda port: agent.build_app(port=port, step_s=0.01)) as url:
        m.research._backend = A2ABackend(url)
        m.research._progress_every_s = None
        async with m.app.test_session(models=models) as s:
            await s.join()
            await s.say("find me a quiet cafe near the station")
            for _ in range(250):
                if len(s.spoken()) == 2:
                    break
                await asyncio.sleep(0.02)
            assert s.spoken() == ["I'm on it.", "I found three options near the station."]
            p = s.participant()
            (job,) = m.research.jobs(p)
            assert job.state == "done"
            assert "Three options for: quiet cafe near the station" in await m.research.results.view(p)
