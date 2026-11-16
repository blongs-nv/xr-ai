# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant task scope and the un-enroll/stop sequences.

The sequences below are the executable form of ``docs/LIFECYCLE_CONTRACT.md`` (in the ``xr-ai-next`` package);
the runtime and the contract suite in :mod:`.testing` both read them so
the written order, the code and the tests cannot drift apart.
"""

from __future__ import annotations

import asyncio
import contextvars
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Final, TypeVar

from loguru import logger

T = TypeVar("T")

UNENROLL_SEQUENCE: Final[tuple[str, ...]] = (
    "stop_admitting_turns",
    "cancel_participant_tasks",
    "drain_participant_tasks",
    "run_unenrolled_hooks_reversed",
    "release_frames_state_artifacts",
)
"""Ordered steps run when one participant is un-enrolled."""

STOP_SEQUENCE: Final[tuple[str, ...]] = (
    "stop_admitting",
    "unenroll_every_participant",
    "run_stop_hooks_reversed",
    "close_model_clients",
    "stop_runtime",
)
"""Ordered steps run when the app stops."""


def _isolated_context() -> contextvars.Context | None:
    """A copy of the current context with its own Relay scope stack (R1.7, R1.8), or ``None`` without Relay.

    A task created in a runtime delivery shares that delivery's Relay stack unless the stack is forked, and a spawned
    task that publishes while the delivery that spawned it is still open then corrupts the stack.

    Forked from a throwaway copy: ``fork_asyncio_context`` first creates a stack in the context it runs in when there
    is none, and that stack would then be shared by every later task copied from the caller's context.
    """
    try:
        import nemo_relay  # noqa: PLC0415 - a dependency of xr-ai-agent-runtime, which this package requires
    except ImportError:  # pragma: no cover
        return None
    return contextvars.copy_context().run(nemo_relay.fork_asyncio_context)


_TaskFactory = Callable[..., "asyncio.Future[Any]"]
_installed: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[_TaskFactory, _TaskFactory | None, int]] = (
    weakref.WeakKeyDictionary()
)


def install_relay_task_factory(loop: asyncio.AbstractEventLoop) -> None:
    """Give every task created on ``loop`` without an explicit context its own Relay scope stack (R1.9).

    ``asyncio.create_task`` copies the context, but Relay's scope stack object is shared by the copy, so two sibling
    tasks that publish, call a model or run a tool at the same time pop each other's scopes and fail. App code that
    uses ``asyncio.TaskGroup`` or ``gather`` on coroutines (and any library that creates tasks) is therefore made
    safe here, once, rather than at each call site. Tasks created with an explicit ``context`` are left alone; a
    factory already installed on the loop still creates the task. Counted per loop, so several apps on one loop
    share one installation.
    """
    if _isolated_context() is None:  # pragma: no cover - Relay is a dependency of the runtime
        return
    current = _installed.get(loop)
    if current is not None:
        factory, previous, count = current
        _installed[loop] = (factory, previous, count + 1)
        return
    previous = loop.get_task_factory()

    def factory(loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, Any], **kwargs: Any) -> asyncio.Future[Any]:
        if kwargs.get("context") is None:
            kwargs["context"] = _isolated_context()
        if previous is not None:
            return previous(loop, coro, **kwargs)
        return asyncio.Task(coro, loop=loop, **kwargs)

    loop.set_task_factory(factory)
    _installed[loop] = (factory, previous, 1)


def uninstall_relay_task_factory(loop: asyncio.AbstractEventLoop) -> None:
    """Undo one :func:`install_relay_task_factory`; the last one restores the loop's earlier factory."""
    current = _installed.get(loop)
    if current is None:
        return
    factory, previous, count = current
    if count > 1:
        _installed[loop] = (factory, previous, count - 1)
        return
    del _installed[loop]
    if loop.get_task_factory() is factory:  # someone else replaced it since: leave theirs in place
        loop.set_task_factory(previous)


def _retrieve(task: asyncio.Task[Any]) -> None:
    """Mark a failed scope task's exception as retrieved: it was already logged and reported (R4.6)."""
    if not task.cancelled():
        task.exception()


@dataclass(frozen=True, slots=True)
class ScopeTimeouts:
    """Bounds on each waiting step of the un-enroll and stop sequences."""

    drain_s: float = 5.0
    """Seconds to wait for a participant's cancelled tasks to finish."""

    hook_s: float = 5.0
    """Seconds each ``on_participant_unenrolled`` or ``on_stop`` hook may run."""

    close_s: float = 10.0
    """Seconds each model client may take to close."""

    start_s: float | None = None
    """Seconds each ``Component.on_start`` hook may run; ``None`` (the default) means ``hook_s``. A long warm-up
    raises this without widening the un-enroll and stop hook budgets (rule R2.7)."""


class ParticipantScope:
    """Owns every task started for one participant.

    ``asyncio.TaskGroup`` cancels every sibling when one child fails, which contradicts rule R4.1
    (an exception stays inside its task). The scope therefore keeps its own task set: each task is
    wrapped so an exception is logged and reported, and cancellation is never reported as an error.
    The task then ends with that exception, so code that awaits it sees the failure (R4.6); nobody
    else does, because the scope marks it retrieved.
    """

    def __init__(self, participant_id: str, *, timeouts: ScopeTimeouts | None = None) -> None:
        self._participant_id = participant_id
        self._timeouts = timeouts or ScopeTimeouts()
        self._admitting = True
        self._tasks: set[asyncio.Task[Any]] = set()
        self._keyed: dict[str, asyncio.Task[Any]] = {}
        self._live: dict[str, set[asyncio.Task[Any]]] = {}
        """Every unfinished task per key, not only the newest: a superseded task may still be cleaning up."""
        self._abandoned: set[asyncio.Task[Any]] = set()
        self._trace: list[str] = []
        self._on_error: Callable[[str | None, BaseException], Awaitable[None]] | None = None
        self._run_hooks: Callable[[], Awaitable[None]] | None = None
        self._release: Callable[[], Awaitable[None]] | None = None
        self._leaving: asyncio.Future[None] | None = None

    def _bind(
        self,
        *,
        on_error: Callable[[str | None, BaseException], Awaitable[None]] | None = None,
        run_hooks: Callable[[], Awaitable[None]] | None = None,
        release: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Attach the app's error reporter, ``on_participant_unenrolled`` runner and resource release."""
        self._on_error = on_error
        self._run_hooks = run_hooks
        self._release = release

    @property
    def participant_id(self) -> str:
        """The participant this scope belongs to."""
        return self._participant_id

    @property
    def admitting(self) -> bool:
        """Whether new turns and spawns are still accepted."""
        return self._admitting

    def spawn(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        key: str | None = None,
        supersede: bool = False,
    ) -> asyncio.Task[T]:
        """Run ``coro`` in this scope; ``supersede`` cancels a live task with the same key first."""
        if not self._admitting:
            coro.close()
            raise RuntimeError(f"participant {self._participant_id!r} scope is no longer admitting tasks")
        previous = {t for t in self._live.get(key, ()) if not t.done()} if key is not None else set()
        if previous and not supersede:
            coro.close()
            raise ValueError(f"a task is already running under key {key!r}")
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._run(coro, key, previous), name=f"{self._participant_id}:{key}", context=_isolated_context()
        )
        self._tasks.add(task)
        task.add_done_callback(lambda _task: coro.close())  # a task cancelled before it started never ran ``coro``
        task.add_done_callback(_retrieve)
        if key is not None:
            self._keyed[key] = task
            self._live.setdefault(key, set()).add(task)
        task.add_done_callback(lambda t: self._forget(t, key))
        return task

    def _forget(self, task: asyncio.Task[Any], key: str | None) -> None:
        self._tasks.discard(task)
        self._abandoned.discard(task)
        if key is not None:
            live = self._live.get(key)
            if live is not None:
                live.discard(task)
                if not live:
                    del self._live[key]
            if self._keyed.get(key) is task:
                del self._keyed[key]

    async def _run(
        self,
        coro: Coroutine[Any, Any, T],
        key: str | None,
        previous: set[asyncio.Task[Any]],
    ) -> T:
        try:
            if previous:
                # Every earlier task under the key, not just the newest, must finish its cleanup first.
                for old in previous:
                    old.cancel()
                _, stragglers = await asyncio.wait(previous, timeout=self._timeouts.drain_s)
                if stragglers:
                    await self._abandon(set(stragglers))
            return await coro
        except asyncio.CancelledError:
            raise
        except (Exception, BaseExceptionGroup) as exc:  # noqa: BLE001 - containment boundary (rule R4)
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise  # this task is being cancelled; a group that carries the cancellation is not an error
            logger.exception("participant {} task {!r} failed", self._participant_id, key)
            await self._report(key, exc)
            raise  # contained: siblings keep running and nobody else sees it, but whoever awaits this task does
        finally:
            coro.close()

    async def _report(self, key: str | None, exc: BaseException) -> None:
        if self._on_error is None:
            return
        try:
            await self._on_error(key, exc)
        except Exception:  # noqa: BLE001 - containment boundary (rule R4)
            logger.exception("participant {} error report failed", self._participant_id)

    async def cancel(self, key: str) -> bool:
        """Cancel and drain the task registered under ``key``; return whether one was live."""
        tasks = {t for t in self._live.get(key, ()) if not t.done()}
        if not tasks:
            return False
        for task in tasks:
            task.cancel()
        tasks.discard(asyncio.current_task())  # type: ignore[arg-type]
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=self._timeouts.drain_s)
            if pending:
                await self._abandon(set(pending))
        return True

    async def _abandon(self, tasks: set[asyncio.Task[Any]]) -> None:
        self._abandoned |= tasks
        for task in tasks:
            logger.error(
                "participant {} task {} ignored cancellation for {}s; abandoned",
                self._participant_id,
                task.get_name(),
                self._timeouts.drain_s,
            )
            await self._report(task.get_name(), TimeoutError("task ignored cancellation and was abandoned"))

    async def leave(self) -> None:
        """Run :data:`UNENROLL_SEQUENCE` for this participant."""
        if self._leaving is not None:
            await asyncio.shield(self._leaving)
            return
        self._leaving = asyncio.get_running_loop().create_future()
        try:
            await self._leave()
        finally:
            self._leaving.set_result(None)

    async def _leave(self) -> None:
        trace = self._trace
        trace.append("stop_admitting_turns")
        self._admitting = False
        trace.append("cancel_participant_tasks")
        me = asyncio.current_task()
        tasks = {t for t in self._tasks if t is not me and not t.done()}
        for task in tasks:
            task.cancel()
        trace.append("drain_participant_tasks")
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=self._timeouts.drain_s)
            if pending:
                await self._abandon(set(pending))
        trace.append("run_unenrolled_hooks_reversed")
        await self._guarded(self._run_hooks, "un-enrolled hooks")
        trace.append("release_frames_state_artifacts")
        await self._guarded(self._release, "release")

    async def _guarded(self, step: Callable[[], Awaitable[None]] | None, label: str) -> None:
        if step is None:
            return
        try:
            await step()
        except Exception:  # noqa: BLE001 - containment boundary (rule R4)
            logger.exception("participant {} leave step {} failed", self._participant_id, label)


__all__ = [
    "UNENROLL_SEQUENCE",
    "STOP_SEQUENCE",
    "ParticipantScope",
    "ScopeTimeouts",
    "install_relay_task_factory",
    "uninstall_relay_task_factory",
]
