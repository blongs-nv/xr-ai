# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegation: long work runs behind the front end as a JOB; the front end stays real time.

A :class:`Delegate` starts jobs on a :class:`Backend` and owns their lifecycle: start, updates, cancel, failure.
Each job's updates become context and (optionally) speech:

- ``delegate.status`` (a session :class:`~xr_ai_app.ContextValue`): what is running, with its latest progress;
- ``delegate.results`` (a session :class:`~xr_ai_app.ContextLog`): finished results, newest last;
- progress is spoken at most every ``progress_every_s``; a result is spoken as one sentence (the backend's
  ``speak``, or a summary by the ``summarize`` role); a failure is spoken.

The backend never speaks; it reports. A backend is any object with ``run(request, job)`` returning an async
iterator of :class:`JobUpdate`: :class:`FunctionBackend` (an async generator function in this process),
:class:`A2ABackend` (a remote agent that speaks the Agent2Agent protocol), or ``xr_ai_kit.langchain.GraphBackend``
(a LangGraph graph).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from xr_ai_app import Component, ContextLog, ContextValue, trace
from xr_ai_app.clock import held

if TYPE_CHECKING:
    from xr_ai_app import Participant, Role, Sink

__all__ = ["A2ABackend", "Backend", "Delegate", "FunctionBackend", "Job", "JobUpdate"]

JobState = Literal["running", "done", "failed", "cancelled"]

_job_numbers = itertools.count(1)


@dataclass(frozen=True, slots=True)
class JobUpdate:
    """One report from a backend: ``progress`` (any number), then one ``result`` or ``error``."""

    kind: Literal["progress", "result", "error"]
    text: str = ""
    speak: str | None = None
    """For a result: one sentence to say. Without it the delegate words one (or says that the result is ready)."""
    data: Any = None


@dataclass(eq=False)
class Job:
    """One delegated request and what became of it."""

    id: str
    request: str
    participant: str
    state: JobState = "running"
    progress: str = ""
    result_text: str = ""
    error: str = ""
    started: float = field(default_factory=time.time)
    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    _ended: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    async def result(self) -> str:
        """Wait for the job to end; the result text (``""`` when it failed or was cancelled)."""
        await self._ended.wait()
        return self.result_text

    def cancel(self) -> bool:
        """Cancel the job; ``False`` when it already ended."""
        if self._task is None or self._task.done():
            return False
        self._task.cancel()
        return True

    def describe(self) -> str:
        age = int(time.time() - self.started)
        line = f"[{self.id}] {self.state}, {age}s: {self.request}"
        if self.state == "running" and self.progress:
            line += f" (latest: {self.progress})"
        if self.state == "failed":
            line += f" (error: {self.error})"
        return line


@runtime_checkable
class Backend(Protocol):
    """Runs one job: yields progress, then a result or an error. Cancellation is task cancellation."""

    def run(self, request: str, job: Job) -> AsyncIterator[JobUpdate]: ...


class FunctionBackend:
    """A backend from an async generator function ``fn(request, job)`` yielding :class:`JobUpdate` (or plain
    strings, which are progress). Whatever it returns last as a ``result`` ends the job."""

    def __init__(self, fn: Callable[[str, Job], AsyncIterator[JobUpdate | str]]) -> None:
        self._fn = fn

    async def run(self, request: str, job: Job) -> AsyncIterator[JobUpdate]:
        async for update in self._fn(request, job):
            yield JobUpdate("progress", update) if isinstance(update, str) else update


class A2ABackend:
    """A remote agent that speaks A2A, the Agent2Agent protocol (https://a2a-protocol.org, spec 1.0).

    ``url`` is the agent's base URL; its agent card (``/.well-known/agent-card.json``) says how to reach it. Each job
    sends the request as one user message and follows the task's stream:

    - the task's status messages while it is submitted or working become ``progress``;
    - its artifacts (text parts, appended chunks joined) become the ``result`` when the task completes; a plain
      message reply with no task is the result;
    - ``failed``, ``rejected`` and ``canceled`` become an ``error``; so do ``input-required`` and ``auth-required``,
      because a delegated job has no way to answer the agent (the task is then cancelled on the agent).

    Cancelling the job (the delegate's ``cancel``, or a new job replacing it) sends ``tasks/cancel`` for a task that
    has not finished, then closes the connection. Resubscribing to a task after a dropped connection is not done.

    Needs the A2A SDK: depend on ``xr-ai-kit[a2a]``. Nothing is fetched or imported until a job runs.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        connect_timeout_s: float = 10.0,
        cancel_timeout_s: float = 5.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._headers = dict(headers or {})
        self._connect_timeout_s = connect_timeout_s
        self._cancel_timeout_s = cancel_timeout_s

    async def run(self, request: str, job: Job) -> AsyncIterator[JobUpdate]:
        try:
            from a2a.client import A2ACardResolver, ClientConfig, create_client  # noqa: PLC0415
            from a2a.helpers import get_artifact_text, get_message_text, new_text_message  # noqa: PLC0415
            from a2a.types import CancelTaskRequest, Role, SendMessageRequest, TaskState  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("A2ABackend needs the A2A SDK: depend on xr-ai-kit[a2a]") from exc
        import httpx  # noqa: PLC0415

        # A stream may be quiet for minutes between updates: only connecting is timed. The A2A client closes this
        # client when it closes, so every job gets its own.
        http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=self._connect_timeout_s), headers=self._headers)
        try:
            card = await A2ACardResolver(http, self._url).get_agent_card()
            client = await create_client(card, ClientConfig(streaming=True, httpx_client=http))
        except BaseException:
            await http.aclose()
            raise
        message = new_text_message(request, role=Role.ROLE_USER)
        message.metadata.update({"job": job.id, "participant": job.participant})
        working = (TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING)
        failed = {
            TaskState.TASK_STATE_FAILED: "the agent failed",
            TaskState.TASK_STATE_REJECTED: "the agent rejected the request",
            TaskState.TASK_STATE_CANCELED: "the agent cancelled the task",
            TaskState.TASK_STATE_INPUT_REQUIRED: "the agent asked for input, which a delegated job cannot give",
            TaskState.TASK_STATE_AUTH_REQUIRED: "the agent asked for authorization, which a delegated job cannot give",
        }
        hold = held()  # the whole job is real external work: a virtual-time loop waits for it for real
        hold.__enter__()
        task_id = ""
        finished = False  # the task reached a terminal state on the agent (nothing left to cancel there)
        artifacts: dict[str, str] = {}
        last_progress = ""
        try:
            async for event in client.send_message(SendMessageRequest(message=message)):
                if event.HasField("message"):  # a direct reply, no task
                    finished = True
                    yield JobUpdate("result", get_message_text(event.message))
                    return
                if event.HasField("task"):
                    task_id = event.task.id
                    status = event.task.status
                    for artifact in event.task.artifacts:
                        artifacts[artifact.artifact_id] = get_artifact_text(artifact)
                elif event.HasField("status_update"):
                    task_id = event.status_update.task_id or task_id
                    status = event.status_update.status
                elif event.HasField("artifact_update"):
                    update = event.artifact_update
                    task_id = update.task_id or task_id
                    text = get_artifact_text(update.artifact)
                    key = update.artifact.artifact_id
                    artifacts[key] = artifacts.get(key, "") + text if update.append else text
                    continue
                else:
                    continue
                said = get_message_text(status.message) if status.HasField("message") else ""
                if status.state in working:
                    if said and said != last_progress:
                        last_progress = said
                        yield JobUpdate("progress", said)
                elif status.state == TaskState.TASK_STATE_COMPLETED:
                    finished = True
                    yield JobUpdate("result", "\n".join(t for t in artifacts.values() if t) or said)
                    return
                elif status.state in failed:
                    finished = status.state not in (
                        TaskState.TASK_STATE_INPUT_REQUIRED,
                        TaskState.TASK_STATE_AUTH_REQUIRED,
                    )
                    yield JobUpdate("error", f"{failed[status.state]}: {said}" if said else failed[status.state])
                    return
            yield JobUpdate("error", "the agent's stream ended before the task finished")
        finally:
            if task_id and not finished:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(client.cancel_task(CancelTaskRequest(id=task_id)), self._cancel_timeout_s)
            await client.close()
            hold.__exit__(None, None, None)


_RESULT_PROMPT = (
    "Say in one short spoken sentence what this result found, for someone who asked: {request}\n\nResult:\n{text}"
)


class Delegate(Component):
    """Runs long requests as jobs on ``backend`` (see the module docstring)."""

    def __init__(
        self,
        name: str,
        *,
        backend: Backend,
        output: Sink | None = None,
        summarize: Role | None = None,
        progress_every_s: float | None = 20.0,
        max_running: int = 1,
        status_budget_bytes: int = 800,
        results_budget_bytes: int = 4_000,
    ) -> None:
        super().__init__()
        if max_running < 1:
            raise ValueError("max_running must be at least 1")
        self._name = name
        self._backend = backend
        self._output = output
        self._summarize = summarize
        self._progress_every_s = progress_every_s
        self._max_running = max_running
        self.status = ContextValue(f"{name} jobs", scope="session", budget_bytes=status_budget_bytes)
        self.results = ContextLog(
            f"{name} results", scope="session", budget_bytes=results_budget_bytes, summarize=summarize
        )

    @property
    def name(self) -> str:
        return self._name

    def jobs(self, p: Participant) -> list[Job]:
        """This participant's jobs in this session, oldest first (running and ended)."""
        return list(self._store(p).values())

    def running(self, p: Participant) -> list[Job]:
        return [j for j in self.jobs(p) if j.state == "running"]

    async def start(self, p: Participant, request: str) -> Job:
        """Start a job; at ``max_running`` the oldest running job is cancelled first. Returns at once."""
        running = self.running(p)
        while len(running) >= self._max_running:
            oldest = running.pop(0)
            oldest.cancel()
            await oldest.result()
        job = Job(id=f"j{next(_job_numbers)}", request=request, participant=p.id)
        self._store(p)[job.id] = job
        job._task = p.spawn(self._run(p, job), key=f"delegate:{self._name}:{job.id}")
        await self._publish_status(p)
        return job

    async def cancel(self, p: Participant, job_id: str | None = None) -> bool:
        """Cancel one job (by id) or every running one; whether anything was running."""
        targets = [j for j in self.running(p) if job_id is None or j.id == job_id]
        for job in targets:
            job.cancel()
        for job in targets:
            await job.result()
        return bool(targets)

    # ------------------------------------------------------------------ internals

    def _store(self, p: Participant) -> dict[str, Job]:
        return p.session.state.setdefault(f"_delegate:{self._name}", {})

    async def _publish_status(self, p: Participant) -> None:
        jobs = [j for j in self.jobs(p) if j.state == "running"] + [
            j for j in self.jobs(p)[-3:] if j.state != "running"
        ]
        if jobs:
            await self.status.set(p, "\n".join(j.describe() for j in jobs))
        else:
            await self.status.clear(p)

    async def _say(self, p: Participant, text: str) -> None:
        if self._output is not None and text:
            await p.say(text, to=self._output, source=f"delegate:{self._name}")

    async def _run(self, p: Participant, job: Job) -> None:
        trace.emit("job", self._name, phase="start", job=job.id, request=job.request)
        last_spoken = asyncio.get_running_loop().time()
        try:
            async for update in self._backend.run(job.request, job):
                if update.kind == "progress":
                    job.progress = update.text
                    trace.emit("job", self._name, phase="progress", job=job.id, text=update.text)
                    await self._publish_status(p)
                    every = self._progress_every_s
                    now = asyncio.get_running_loop().time()
                    if every is not None and now - last_spoken >= every:
                        last_spoken = now
                        await self._say(p, f"Still working on it: {update.text}.")
                elif update.kind == "result":
                    job.state, job.result_text = "done", update.text
                    break
                else:
                    job.state, job.error = "failed", update.text or "the back end reported an error"
                    break
            else:
                if job.state == "running":
                    job.state, job.error = "failed", "the back end ended without a result"
        except asyncio.CancelledError:
            job.state = "cancelled"
            trace.emit("job", self._name, phase="cancelled", job=job.id)
            with contextlib.suppress(Exception):
                await self._publish_status(p)
            job._ended.set()
            raise
        except Exception as exc:  # noqa: BLE001 - a failed back end is reported, not raised
            job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
        trace.emit("job", self._name, phase=job.state, job=job.id, error=job.error or None)
        try:
            if job.state == "done":
                await self.results.publish(p, f"[{job.id}] {job.request}\n{job.result_text}")
                await self._publish_status(p)
                await self._say(p, await self._spoken_result(job, update.speak))
            else:
                await self._publish_status(p)
                await self._say(p, f"Sorry, the {self._name} task failed.")
        finally:
            job._ended.set()

    async def _spoken_result(self, job: Job, speak: str | None) -> str:
        if speak:
            return speak
        if self._summarize is not None:
            with contextlib.suppress(Exception):
                text = await self._summarize.ask(
                    None,
                    _RESULT_PROMPT.format(request=job.request, text=job.result_text[:4000]),
                    thinking=False,
                    max_tokens=80,
                )
                if str(text).strip():
                    return str(text).strip()
        return f"Your {self._name} result is ready."
