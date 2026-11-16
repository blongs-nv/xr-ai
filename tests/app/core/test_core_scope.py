# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant task scope: ownership, supersede, containment and the un-enroll sequence (rules R1, R2, R4)."""

from __future__ import annotations

import asyncio

import pytest
from xr_ai_app._base.scope import UNENROLL_SEQUENCE, ParticipantScope, ScopeTimeouts


def _scope(**timeouts: float) -> tuple[ParticipantScope, list[tuple[str | None, str]]]:
    scope = ParticipantScope("p1", timeouts=ScopeTimeouts(**timeouts) if timeouts else None)
    errors: list[tuple[str | None, str]] = []

    async def on_error(key: str | None, exc: BaseException) -> None:
        errors.append((key, type(exc).__name__))

    scope._bind(on_error=on_error)
    return scope, errors


async def test_spawn_runs_the_coroutine_and_returns_its_task() -> None:
    scope, _ = _scope()

    async def work() -> int:
        return 7

    task = scope.spawn(work())
    assert isinstance(task, asyncio.Task)
    assert await task == 7


async def test_supersede_cancels_and_drains_the_old_task_before_the_new_one_starts() -> None:
    scope, _ = _scope()
    log: list[str] = []

    async def first() -> None:
        log.append("first:start")
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0.02)  # cleanup that takes time must finish before the next turn
            log.append("first:cleaned")

    async def second() -> None:
        log.append("second:start")

    scope.spawn(first(), key="turn", supersede=True)
    await asyncio.sleep(0)
    await scope.spawn(second(), key="turn", supersede=True)
    assert log == ["first:start", "first:cleaned", "second:start"]


async def test_spawn_under_a_live_key_without_supersede_raises_and_never_runs() -> None:
    scope, _ = _scope()
    started: list[str] = []

    async def blocked() -> None:
        await asyncio.sleep(10)

    async def duplicate() -> None:
        started.append("ran")

    scope.spawn(blocked(), key="k")
    coro = duplicate()
    with pytest.raises(ValueError, match="'k'"):
        scope.spawn(coro, key="k")
    assert started == []
    assert await scope.cancel("k") is True


async def test_cancel_returns_whether_a_task_was_live() -> None:
    scope, errors = _scope()

    async def blocked() -> None:
        await asyncio.sleep(10)

    task = scope.spawn(blocked(), key="k")
    await asyncio.sleep(0)
    assert await scope.cancel("k") is True
    assert task.cancelled()
    assert await scope.cancel("k") is False
    assert await scope.cancel("never") is False
    assert errors == []  # cancellation is not an error (R4.5)


async def test_failure_is_contained_reported_and_leaves_siblings_running() -> None:
    scope, errors = _scope()
    sibling_done = asyncio.Event()

    async def boom() -> None:
        raise RuntimeError("bad")

    async def sibling() -> None:
        await asyncio.sleep(0.02)
        sibling_done.set()

    scope.spawn(sibling(), key="sibling")
    failing = scope.spawn(boom(), key="boom")
    await asyncio.wait({failing})
    await asyncio.wait_for(sibling_done.wait(), 1)
    assert errors == [("boom", "RuntimeError")]


async def test_an_awaiting_caller_sees_the_failure_after_it_was_reported() -> None:
    """R4.6: the failure is logged and reported once, then whoever awaits the task gets the exception."""
    scope, errors = _scope()

    async def boom() -> None:
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await scope.spawn(boom(), key="boom")
    assert errors == [("boom", "ValueError")]


async def test_an_unawaited_failure_is_never_reported_as_unretrieved() -> None:
    """R4.6: a fire-and-forget task that fails does not trigger asyncio's 'never retrieved' handler."""
    scope, _ = _scope()
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, object]] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    async def boom() -> None:
        raise RuntimeError("bad")

    try:
        task = scope.spawn(boom(), key="boom")
        await asyncio.wait({task})
        del task
        import gc  # noqa: PLC0415 - only this test needs a collection

        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)
    assert unhandled == []


async def test_leave_runs_the_five_steps_in_order_and_hooks_see_a_closed_scope() -> None:
    scope, _ = _scope()
    log: list[str] = []

    async def hooks() -> None:
        log.append(f"hooks:admitting={scope.admitting}:tasks={len(scope._tasks)}")

    async def release() -> None:
        log.append("release")

    scope._bind(run_hooks=hooks, release=release)

    async def blocked() -> None:
        await asyncio.sleep(10)

    tasks = [scope.spawn(blocked(), key=str(i)) for i in range(3)]
    await asyncio.sleep(0)
    await scope.leave()
    assert scope._trace == list(UNENROLL_SEQUENCE)
    assert all(t.cancelled() for t in tasks)
    assert log == ["hooks:admitting=False:tasks=0", "release"]


async def test_spawn_after_leave_raises_and_does_not_start_the_coroutine() -> None:
    scope, _ = _scope()
    await scope.leave()
    ran: list[str] = []

    async def late() -> None:
        ran.append("ran")

    with pytest.raises(RuntimeError, match="no longer admitting"):
        scope.spawn(late())
    await asyncio.sleep(0)
    assert ran == []
    assert scope.admitting is False


async def test_a_task_that_ignores_cancellation_is_abandoned_after_drain_s_and_leave_continues() -> None:
    scope, errors = _scope(drain_s=0.05)
    released: list[str] = []
    stubborn_may_exit = asyncio.Event()

    async def stubborn() -> None:
        while not stubborn_may_exit.is_set():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                continue  # swallows cancellation

    async def release() -> None:
        released.append("release")

    scope._bind(
        on_error=lambda key, exc: _record(errors, key, exc),
        release=release,
    )
    scope.spawn(stubborn(), key="stubborn")
    await asyncio.sleep(0)
    await asyncio.wait_for(scope.leave(), 1)
    assert released == ["release"]
    assert errors == [("p1:stubborn", "TimeoutError")]
    stubborn_may_exit.set()
    for task in tuple(scope._abandoned):
        task.cancel()


async def _record(errors: list[tuple[str | None, str]], key: str | None, exc: BaseException) -> None:
    errors.append((key, type(exc).__name__))


async def test_leave_is_idempotent_and_concurrent_calls_share_one_sequence() -> None:
    scope, _ = _scope()
    calls: list[str] = []

    async def hooks() -> None:
        calls.append("hooks")
        await asyncio.sleep(0.02)

    scope._bind(run_hooks=hooks)
    await asyncio.gather(scope.leave(), scope.leave())
    await scope.leave()
    assert calls == ["hooks"]
    assert scope._trace == list(UNENROLL_SEQUENCE)


async def test_a_failing_hook_runner_does_not_stop_the_release_step() -> None:
    scope, _ = _scope()
    released: list[str] = []

    async def hooks() -> None:
        raise RuntimeError("hook failed")

    async def release() -> None:
        released.append("release")

    scope._bind(run_hooks=hooks, release=release)
    await scope.leave()
    assert released == ["release"]


async def test_three_back_to_back_supersedes_wait_for_every_earlier_cleanup() -> None:
    scope, _ = _scope()
    log: list[str] = []

    async def first() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0.02)
            log.append("first:cleaned")

    async def middle() -> None:
        log.append("middle:start")

    async def last() -> None:
        log.append("last:start")

    scope.spawn(first(), key="turn", supersede=True)
    await asyncio.sleep(0)
    scope.spawn(middle(), key="turn", supersede=True)  # no await between the next two spawns
    task = scope.spawn(last(), key="turn", supersede=True)
    await task
    assert log.index("first:cleaned") < log.index("last:start")


async def test_a_predecessor_that_ignores_cancellation_delays_the_next_turn_only_until_drain_s() -> None:
    scope, errors = _scope(drain_s=0.05)
    release = asyncio.Event()

    async def stubborn() -> None:
        while not release.is_set():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                continue

    async def next_turn() -> str:
        return "ran"

    scope.spawn(stubborn(), key="turn", supersede=True)
    await asyncio.sleep(0)
    result = await asyncio.wait_for(scope.spawn(next_turn(), key="turn", supersede=True), 1)
    assert result == "ran"
    assert errors and errors[0][1] == "TimeoutError"
    release.set()
    for task in tuple(scope._abandoned):
        task.cancel()


async def test_cancel_drains_every_task_under_the_key() -> None:
    scope, _ = _scope()
    cleaned: list[str] = []

    async def worker(name: str) -> None:
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0.01)
            cleaned.append(name)

    scope.spawn(worker("a"), key="k", supersede=True)
    await asyncio.sleep(0)
    scope.spawn(worker("b"), key="k", supersede=True)
    scope.spawn(worker("c"), key="k", supersede=True)  # b and c never start: a is still draining
    await asyncio.sleep(0)
    assert len(scope._live["k"]) == 3
    assert await scope.cancel("k") is True
    assert cleaned == ["a"] and scope._live == {}
