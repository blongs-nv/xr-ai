# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lifecycle contract (the package's ``docs/LIFECYCLE_CONTRACT.md``) as a reusable suite.

``await run_lifecycle_contract(make_app)`` runs one check per rule in :data:`LIFECYCLE_RULES`
against apps built by ``make_app``. Each check builds a fresh app, adds probe components with
``app.add``, drives it through ``test_session`` and asserts the observable facts of the rule.
``make_app`` must return an app with no behavior of its own that conflicts with the probes.

Limits, so a green run is read correctly:

- R2.6 (a task that ignores cancellation) needs ``app._config.timeouts.drain_s <= 2.0``; with a longer
  bound the sub-check is skipped and a ``UserWarning`` says so.
- Statements about turns and tools (R1.5, R2.2 waiting turns, R5.2, R6.3) belong to
  the WPs that implement those features and are covered there; R5 is checked through a stream
  started from a spawned task, which is the same scope that runs turns.
"""

from __future__ import annotations

import asyncio
import signal
import warnings
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Final

from loguru import logger
from xr_ai_runtime import RuntimeClosedError

from ..component import Component
from ._models import FakeModels
from ._session import test_session

if TYPE_CHECKING:
    from ..app import App
    from ..participant import Participant

LIFECYCLE_RULES: Final[tuple[str, ...]] = (
    "R1_task_ownership",
    "R2_unenroll_sequence",
    "R3_stop_sequence",
    "R4_error_containment",
    "R5_cancel_turn_cancels_model_stream",
    "R6_no_cancelled_or_closed_errors_in_app_code",
    "R7_only_run_installs_signal_handlers",
    "R8_composition",
)
"""One identifier per rule in the package's ``docs/LIFECYCLE_CONTRACT.md``; the suite has one check per entry."""

PENDING_RULES: Final[frozenset[str]] = frozenset({"R8_composition"})
"""Rules whose check is a placeholder (epic C: the ``Sink``, ``Hub``, ``Voice``, ``Conversation`` and ``Router``
components). ``run_lifecycle_rule`` raises ``NotImplementedError`` for them (tests mark that an expected
failure); ``run_lifecycle_contract`` skips them silently (debug log). Supplying the last piece empties
this set."""

_Hook = Callable[["Participant"], Awaitable[None]]
_NEVER = 3600.0


def _expect(condition: object, rule: str, message: str) -> None:
    if not condition:
        raise AssertionError(f"{rule}: {message}")


class _Log(list[str]):
    """Hook trace. App code contains hook errors (R2.7), so assertion failures raised inside a
    probe hook are collected here and re-raised by the check once the session is over."""

    def __init__(self) -> None:
        super().__init__()
        self.failures: list[AssertionError] = []

    def raise_failures(self) -> None:
        if self.failures:
            raise self.failures[0]


class _Probe(Component):
    """A component that records its hook calls into a shared log and runs optional callbacks."""

    def __init__(
        self,
        name: str,
        log: _Log,
        *,
        joined: _Hook | None = None,
        left: _Hook | None = None,
        start_error: Exception | None = None,
        left_error: Exception | None = None,
    ) -> None:
        self._name = name
        self._log = log
        self._joined = joined
        self._left = left
        self._start_error = start_error
        self._left_error = left_error

    async def on_start(self) -> None:
        self._log.append(f"start:{self._name}")
        if self._start_error is not None:
            raise self._start_error

    async def on_participant_enrolled(self, p: Participant) -> None:
        self._log.append(f"joined:{self._name}")
        if self._joined is not None:
            try:
                await self._joined(p)
            except AssertionError as exc:
                self._log.failures.append(exc)

    async def on_participant_unenrolled(self, p: Participant) -> None:
        self._log.append(f"left:{self._name}")
        if self._left is not None:
            try:
                await self._left(p)
            except AssertionError as exc:
                self._log.failures.append(exc)
        if self._left_error is not None:
            raise self._left_error

    async def on_stop(self) -> None:
        self._log.append(f"stop:{self._name}")


async def _blocker(log: _Log, tag: str) -> None:
    log.append(f"{tag}:start")
    try:
        await asyncio.sleep(_NEVER)
    except asyncio.CancelledError:
        log.append(f"{tag}:cancelled")
        raise


async def _marker(log: _Log, tag: str) -> None:
    log.append(f"{tag}:start")


async def _boom() -> None:
    raise ValueError("probe task failure")


def _spawn_rejected(p: Participant, log: _Log, exc: type[Exception], rule: str) -> None:
    coro = _marker(log, "rejected")
    try:
        p.spawn(coro, key="rejected")
    except exc:
        return
    finally:
        coro.close()
    raise AssertionError(f"{rule}: spawn did not raise {exc.__name__}")


# -- R1 ------------------------------------------------------------------------------------


async def _check_r1(make_app: Callable[[], App]) -> None:
    rule = "R1_task_ownership"
    app, log = make_app(), _Log()

    async def joined(p: Participant) -> None:
        task = p.spawn(_blocker(log, "a"), key="a")
        _expect(isinstance(task, asyncio.Task), rule, "R1.1 spawn must return an asyncio.Task")

        p.spawn(_blocker(log, "first"), key="k")
        await asyncio.sleep(0)
        _expect("first:start" in log, rule, "R1.2 setup: first task should have started")
        p.spawn(_marker(log, "second"), key="k", supersede=True)
        await asyncio.sleep(0.05)
        _expect("first:cancelled" in log, rule, "R1.2 supersede must cancel the live task")
        _expect(
            "second:start" in log and log.index("first:cancelled") < log.index("second:start"),
            rule,
            "R1.2 the superseded task must be cancelled before the new one starts",
        )

        coro = _marker(log, "dup")
        try:
            p.spawn(coro, key="a")
        except ValueError:
            pass
        else:
            raise AssertionError(f"{rule}: R1.2 spawn on a live key without supersede must raise ValueError")
        finally:
            coro.close()
        _expect("dup:start" not in log, rule, "R1.2 a rejected spawn must not start the coroutine")

        _expect(await p.cancel("a") is True, rule, "R1.3 cancel of a live task must return True")
        _expect(task.done(), rule, "R1.3 cancel must wait for the task to finish")
        _expect(await p.cancel("a") is False, rule, "R1.3 cancel with no live task must return False")

    async def left(p: Participant) -> None:
        _spawn_rejected(p, log, RuntimeError, f"{rule} R1.4")
        _expect("rejected:start" not in log, rule, "R1.4 a rejected spawn must never start the coroutine")

    app.add(_Probe("A", log, joined=joined, left=left))
    async with test_session(app) as s:
        await s.join()
        await s.settle()
        await s.leave()
        await s.settle()
    log.raise_failures()


# -- R2 ------------------------------------------------------------------------------------


async def _check_r2(make_app: Callable[[], App]) -> None:
    rule = "R2_unenroll_sequence"
    app, log = make_app(), _Log()
    tasks: dict[str, asyncio.Task[Any]] = {}
    fresh: list[bool] = []

    async def joined(p: Participant) -> None:
        fresh.append("marker" not in p.state)
        p.state["marker"] = p.id
        tasks[p.id] = p.spawn(_blocker(log, f"turn-{p.id}"), key="turn")

    async def left(p: Participant) -> None:
        if p.id != "p1":
            return
        _expect(
            "turn-p1:cancelled" in log and tasks["p1"].done(),
            rule,
            "R2.1 participant tasks must be cancelled and drained before on_participant_unenrolled runs",
        )
        _spawn_rejected(p, log, RuntimeError, f"{rule} R2.1 step 1")
        _expect(p.state.get("marker") == "p1", rule, "R2.3 hooks must see p.state intact")
        await p.say("late")
        _expect("late" not in session.spoken("p1"), rule, "R2.3/R6.1 output after step 1 is dropped")

    app.add(_Probe("A", log, joined=joined, left=left))
    app.add(_Probe("B", log, left=None))
    async with test_session(app) as session:
        await session.join("p1")
        await session.join("p2")
        await session.settle()
        await session.leave("p1")
        await session.settle()
        _expect(
            log.index("left:B") < log.index("left:A"),
            rule,
            "R2.1 on_participant_unenrolled must run in reverse registration order",
        )
        _expect(not tasks["p2"].done(), rule, "R2.5 leaving p1 must not cancel p2's tasks")
        _expect("turn-p2:cancelled" not in log, rule, "R2.5 leaving p1 must not cancel p2's tasks")

        await session.join("p1")
        await session.settle()
        _expect(len(fresh) == 3 and all(fresh), rule, "R2.4 a rejoined participant must start with fresh state")
        _expect(
            log.count("turn-p1:cancelled") == 1 and not tasks["p1"].done(),
            rule,
            "R2.4 rejoin must start a fresh scope with no stale tasks",
        )
        await session.leave("p1")
        await session.leave("p2")
    log.raise_failures()

    await _check_r2_hook_errors(make_app)
    await _check_r2_abandon(make_app)


async def _check_r2_hook_errors(make_app: Callable[[], App]) -> None:
    rule = "R2_unenroll_sequence"
    app, log = make_app(), _Log()
    app.add(_Probe("A", log))
    app.add(_Probe("B", log, left_error=RuntimeError("hook failed")))
    app.add(_Probe("C", log))
    async with test_session(app) as s:
        await s.join()
        await s.settle()
        await s.leave()
        await s.settle()
        _expect(
            [e for e in log if e.startswith("left:")] == ["left:C", "left:B", "left:A"],
            rule,
            "R2.7 a failing hook must not stop the remaining hooks",
        )


async def _check_r2_abandon(make_app: Callable[[], App]) -> None:
    rule = "R2_unenroll_sequence"
    app, log = make_app(), _Log()
    drain_s = app._config.timeouts.drain_s
    if drain_s > 2.0:
        warnings.warn(f"R2.6 skipped: drain_s={drain_s} exceeds 2.0; use a shorter drain_s", stacklevel=2)
        return
    release = asyncio.Event()

    async def stubborn() -> None:
        while not release.is_set():
            try:
                await asyncio.wait_for(release.wait(), _NEVER)
            except asyncio.CancelledError:
                log.append("stubborn:ignored-cancel")

    async def joined(p: Participant) -> None:
        p.spawn(stubborn(), key="stubborn")

    app.add(_Probe("A", log, joined=joined))
    try:
        async with test_session(app) as s:
            await s.join()
            await s.settle()
            started = asyncio.get_running_loop().time()
            await asyncio.wait_for(s.leave(), drain_s + app._config.timeouts.hook_s + 3.0)
            _expect(
                "left:A" in log,
                rule,
                "R2.6 the un-enroll sequence must continue after abandoning a task that ignores cancellation",
            )
            _expect(
                asyncio.get_running_loop().time() - started >= drain_s * 0.9,
                rule,
                "R2.6 the task must be given drain_s before it is abandoned",
            )
            _expect(s.hub.statuses("p1"), rule, "R2.6 the abandoned task must be reported on the participant's status")
            release.set()
        log.raise_failures()
    finally:
        release.set()


# -- R3 ------------------------------------------------------------------------------------


async def _check_r3(make_app: Callable[[], App]) -> None:
    rule = "R3_stop_sequence"
    app, log, errors = make_app(), _Log(), []
    before = set(asyncio.all_tasks())

    async def chatter(p: Participant) -> None:
        try:
            while True:
                await p.say("tick")
                await asyncio.sleep(0)
        except RuntimeClosedError as exc:
            errors.append(exc)
            raise

    async def joined(p: Participant) -> None:
        p.spawn(chatter(p), key="chatter")
        p.spawn(_blocker(log, "turn"), key="turn")

    app.add(_Probe("A", log, joined=joined))
    app.add(_Probe("B", log))
    async with test_session(app) as s:
        _expect(
            log[:2] == ["start:A", "start:B"],
            rule,
            "R3.3 on_start hooks must run in registration order before any participant joins",
        )
        await s.join()
        await s.settle()
    _expect(
        [e for e in log if e.split(":")[0] in {"left", "stop"}] == ["left:B", "left:A", "stop:B", "stop:A"],
        rule,
        "R3.1 stop must leave every participant (hooks reversed) and then run on_stop hooks reversed",
    )
    _expect(not errors, rule, "R3.2 no task may publish after the runtime stopped (RuntimeClosedError seen)")
    await asyncio.gather(app.stop(), app.stop())
    _expect(log.count("stop:A") == 1, rule, "R3.4 a second stop() must not run the sequence again")
    await asyncio.sleep(0)
    leftover = {t for t in asyncio.all_tasks() - before if not t.done()} - {asyncio.current_task()}
    _expect(not leftover, rule, f"R3 no task may survive stop(); still running: {sorted(map(repr, leftover))}")

    log2 = _Log()
    app2 = make_app()
    app2.add(_Probe("A", log2))
    app2.add(_Probe("B", log2, start_error=RuntimeError("start failed")))
    try:
        async with test_session(app2):
            raise AssertionError(f"{rule}: R3.3 a failing on_start must make start() raise")
    except RuntimeError as exc:
        _expect(str(exc) == "start failed", rule, "R3.3 start() must re-raise the on_start error")
    _expect("stop:A" in log2, rule, "R3.3 start() must run the stop sequence for what already started")
    log.raise_failures()


# -- R4 ------------------------------------------------------------------------------------


def _error_statuses(s: Any, pid: str) -> int:
    return sum(1 for text, _ in s.hub.statuses(pid) if text == "error")


async def _check_r4(make_app: Callable[[], App]) -> None:
    rule = "R4_error_containment"
    app, log = make_app(), _Log()
    tasks: dict[str, asyncio.Task[Any]] = {}

    async def joined(p: Participant) -> None:
        tasks[f"sib-{p.id}"] = p.spawn(_blocker(log, f"sib-{p.id}"), key="sib")
        if p.id == "p1":
            tasks["boom"] = p.spawn(_boom(), key="boom")

    app.add(_Probe("A", log, joined=joined))
    async with test_session(app) as s:
        await s.join("p1")
        await s.join("p2")
        await s.settle()
        await asyncio.sleep(0.05)
        _expect(_error_statuses(s, "p1") >= 1, rule, "R4.1 a failing task must publish status 'error'")
        _expect(not tasks["sib-p1"].done(), rule, "R4.1 sibling tasks of the same participant keep running")
        _expect(not tasks["sib-p2"].done(), rule, "R4.2 an error in one participant must not touch another")
        _expect(_error_statuses(s, "p2") == 0, rule, "R4.2 another participant's status must stay clean")
        errors_before = _error_statuses(s, "p1")
        await s.leave("p1")
        await s.settle()
        _expect(
            _error_statuses(s, "p1") == errors_before,
            rule,
            "R4.5 cancellation must not be reported as an error",
        )


# -- R5 ------------------------------------------------------------------------------------


async def _consume(app: App) -> None:
    async for _ in app.role("answer").stream(None, "tell me a story"):
        pass


async def _check_r5(make_app: Callable[[], App]) -> None:
    rule = "R5_cancel_turn_cancels_model_stream"
    app = make_app()
    models = FakeModels({"answer": ["once upon a time", "there was a kettle"]})
    models.set_latency("answer", 0.0, chunk_s=_NEVER)
    log = _Log()

    refs: dict[str, Participant] = {}

    async def joined(p: Participant) -> None:
        refs["p"] = p
        p.spawn(_consume(app), key="turn")

    app.add(_Probe("A", log, joined=joined))
    async with test_session(app, models=models) as s:
        await s.join()
        await models.wait_started("answer", 1)
        calls = models.calls("answer")
        await refs["p"].cancel("turn")
        _expect(
            calls[0].cancelled and calls[0].close_count == 1,
            rule,
            "R5.1/R5.3 cancelling a turn's task must close its model stream exactly once",
        )
        refs["p"].spawn(_consume(app), key="turn")
        await models.wait_started("answer", 2)
        await s.leave()
        await s.settle()
        _expect(
            all(c.cancelled and c.close_count == 1 for c in models.calls("answer")),
            rule,
            "R5.1/R5.3 a participant leaving must close every in-flight model stream exactly once",
        )
        _expect(models.active("answer") == 0, rule, "R5.1 no model call may outlive the un-enroll sequence")


# -- R6 ------------------------------------------------------------------------------------


async def _check_r6(make_app: Callable[[], App]) -> None:
    rule = "R6_no_cancelled_or_closed_errors_in_app_code"
    app, log = make_app(), _Log()
    refs: dict[str, Participant] = {}

    async def joined(p: Participant) -> None:
        refs["p"] = p

    async def left(p: Participant) -> None:
        await p.say("late-say")
        await p.reply("late-reply")

    app.add(_Probe("A", log, joined=joined, left=left))
    async with test_session(app) as s:
        await s.join()
        await s.settle()
        await s.leave()
        await s.settle()
        spoken = s.spoken()
        _expect(
            "late-say" not in spoken and "late-reply" not in spoken,
            rule,
            "R6.1 output after the scope stopped admitting must be a silent no-op",
        )
    await refs["p"].say("after-stop")
    _expect("after-stop" not in s.spoken(), rule, "R6.1 output after the app stopped must be a silent no-op")
    log.raise_failures()


# -- R7 ------------------------------------------------------------------------------------


def _signal_state() -> tuple[Any, ...]:
    loop = asyncio.get_running_loop()
    handlers = dict(getattr(loop, "_signal_handlers", {}))
    return (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM), handlers)


async def _check_r7(make_app: Callable[[], App]) -> None:
    rule = "R7_only_run_installs_signal_handlers"
    app, log = make_app(), _Log()
    app.add(_Probe("A", log))
    before = _signal_state()
    async with test_session(app) as s:
        _expect(_signal_state() == before, rule, "R7.1 start() and test_session() must install no signal handler")
        await s.join()
        await s.leave()
    _expect(_signal_state() == before, rule, "R7.1 stop() must leave signal handlers as they were")


async def _check_r8(make_app: Callable[[], App]) -> None:
    raise NotImplementedError("R8_composition: pending until the components exist")


_CHECKS: Final[dict[str, Callable[[Callable[[], App]], Awaitable[None]]]] = {
    "R1_task_ownership": _check_r1,
    "R2_unenroll_sequence": _check_r2,
    "R3_stop_sequence": _check_r3,
    "R4_error_containment": _check_r4,
    "R5_cancel_turn_cancels_model_stream": _check_r5,
    "R6_no_cancelled_or_closed_errors_in_app_code": _check_r6,
    "R7_only_run_installs_signal_handlers": _check_r7,
    "R8_composition": _check_r8,
}


async def run_lifecycle_rule(rule: str, make_app: Callable[[], App]) -> None:
    """Run the single check for ``rule`` (one of :data:`LIFECYCLE_RULES`) against apps from ``make_app``."""
    await _CHECKS[rule](make_app)


async def run_lifecycle_contract(make_app: Callable[[], App]) -> None:
    """Run every check in :data:`LIFECYCLE_RULES` against apps built by ``make_app``.

    Raises ``AssertionError`` listing every rule that failed. Any other exception (for example
    ``NotImplementedError`` from a skeleton) propagates unchanged.
    Rules in :data:`PENDING_RULES` are skipped silently.
    """
    failures: list[str] = []
    for rule in LIFECYCLE_RULES:
        if rule in PENDING_RULES:
            logger.debug("{}: check is a pending placeholder, skipped", rule)
            continue
        try:
            await run_lifecycle_rule(rule, make_app)
        except AssertionError as exc:
            failures.append(str(exc) or rule)
    if failures:
        raise AssertionError("lifecycle contract failed:\n  " + "\n  ".join(failures))
