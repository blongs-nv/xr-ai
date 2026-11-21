# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``App``: config and profile loading, model roles, components, turns, tasks, the trace, signals and shutdown."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from loguru import logger
from pydantic import BaseModel, ValidationError
from xr_ai_interaction import Interrupted, ResponseOutput, UserQuery
from xr_ai_runtime import Agent as RuntimeAgent
from xr_ai_runtime import AgentRuntime, RuntimeClosedError, RuntimeContext, subscribe

from . import _ready, _tracing
from ._bound import ClientPool
from .artifacts import ArtifactSink
from .component import Component
from .enrollment import Connection, Enrollment, PerConnection
from .participant import CURRENT_TURN, Participant, ParticipantState, Turn
from .profiles import PROFILE_ENV, Profile, Role, _category, load_profile, select_profile
from .scope import (
    STOP_SEQUENCE,
    ParticipantScope,
    ScopeTimeouts,
    install_relay_task_factory,
    uninstall_relay_task_factory,
)
from .topics import (
    INTERRUPTED,
    OUTPUT,
    PARTICIPANT_CONNECTED,
    PARTICIPANT_DISCONNECTED,
    USER_QUERY,
    ParticipantConnected,
    ParticipantDisconnected,
)

if TYPE_CHECKING:
    from xr_ai_voice._processors import VadConfig

    from .testing import FakeHub, FakeModels, TestSession

C = TypeVar("C", bound=Component)
F = TypeVar("F", bound=Callable[..., Awaitable[Any]])
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    """Voice settings; replaces ``VadConfig`` and the private voice-gate fields."""

    enabled: bool = True
    """Whether the app runs a voice session; ``false`` runs it on typed input only."""

    vad: VadConfig | None = None
    """Voice-activity settings; ``None`` uses the voice package defaults."""

    voice_gate_yaml: Path | None = None
    """Voice-gate file; the speech models themselves come from the ``speech_in``/``speech_out`` roles."""

    idle_timeout_s: float | None = None
    """Seconds without activity after which the voice pipeline's idle timeout fires; ``None`` disables it."""

    interrupt_on_supersede: bool = True
    """Whether a newer reply cuts the speech of the one it supersedes (the app's behavior so far)."""


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Parsed ``app.yaml``."""

    config_dir: Path
    """Directory of the config file; relative paths resolve against it."""

    profile: str | None = None
    """Default profile named in the file; ``--profile`` and ``XR_AI_PROFILE`` outrank it."""

    voice: VoiceConfig = VoiceConfig()
    """Voice settings."""

    artifacts_dir: Path | None = None
    """Root of the per-participant artifact directories; ``None`` disables ``p.artifacts``."""

    trace: Mapping[str, Any] = field(default_factory=dict)
    """Where trace events go (``trace:`` in ``app.yaml``: ``jsonl``, ``log``, ``relay``); see :attr:`App.trace`."""

    timeouts: ScopeTimeouts = ScopeTimeouts()
    """Bounds on the waiting steps of the leave and stop sequences."""

    settings: Mapping[str, Any] = field(default_factory=dict)
    """The app's own settings, read from the top-level ``settings:`` key; see :meth:`App.settings`."""


_TOP_KEYS = {"profile", "voice", "artifacts_dir", "timeouts", "settings", "trace"}
_VOICE_KEYS = {"enabled", "vad", "voice_gate_yaml", "idle_timeout_s", "interrupt_on_supersede"}


def _path(config_dir: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else config_dir / path


def _load_config(path: str | Path) -> AppConfig:
    """Parse ``app.yaml``; unknown keys are rejected so typos do not pass silently."""
    file = Path(path).expanduser().resolve()
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{file}: top-level must be a mapping")
    if unknown := set(raw) - _TOP_KEYS:
        raise ValueError(f"{file}: unknown keys {sorted(unknown)}")
    config_dir = file.parent
    voice_raw = raw.get("voice") or {}
    if not isinstance(voice_raw, dict) or (unknown := set(voice_raw) - _VOICE_KEYS):
        raise ValueError(f"{file}: voice must be a mapping with keys {sorted(_VOICE_KEYS)}")
    vad = None
    if voice_raw.get("vad") is not None:
        from xr_ai_voice._processors import (
            VadConfig,
        )  # noqa: PLC0415 - keeps the app module free of the voice stack

        vad = VadConfig(**voice_raw["vad"])
    settings = raw.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError(f"{file}: settings must be a mapping")
    timeouts_raw = raw.get("timeouts") or {}
    if not isinstance(timeouts_raw, dict):
        raise ValueError(f"{file}: timeouts must be a mapping")
    if unknown := set(timeouts_raw) - {"drain_s", "hook_s", "close_s", "start_s"}:
        raise ValueError(f"{file}: timeouts accepts drain_s, hook_s, close_s and start_s, not {sorted(unknown)}")
    values: dict[str, float] = {}
    for key, value in timeouts_raw.items():
        if value is None and key == "start_s":
            continue  # ``start_s: null`` means the default (hook_s)
        try:
            values[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{file}: timeouts.{key} must be a number of seconds, got {value!r}") from exc
    timeouts = ScopeTimeouts(**values)
    trace = raw.get("trace") or {}
    if not isinstance(trace, dict) or set(trace) - {"jsonl", "log", "relay"}:
        raise ValueError(f"{file}: trace accepts jsonl, log and relay")
    if trace.get("jsonl") is not None:
        trace = {**trace, "jsonl": _path(config_dir, trace["jsonl"])}
    return AppConfig(
        config_dir=config_dir,
        profile=raw.get("profile"),
        voice=VoiceConfig(
            enabled=bool(voice_raw.get("enabled", True)),
            vad=vad,
            voice_gate_yaml=_path(config_dir, voice_raw.get("voice_gate_yaml")),
            idle_timeout_s=(None if voice_raw.get("idle_timeout_s") is None else float(voice_raw["idle_timeout_s"])),
            interrupt_on_supersede=bool(voice_raw.get("interrupt_on_supersede", True)),
        ),
        artifacts_dir=_path(config_dir, raw.get("artifacts_dir")),
        timeouts=timeouts,
        settings=settings,
        trace=trace,
    )


_NO_WAITER: Any = object()
_warned_no_waiter = False


def _parked(task: asyncio.Task[Any]) -> bool:
    """Whether ``task`` is waiting on an unfinished future (so not runnable), for ``App._is_idle``.

    asyncio has no public way to ask this. CPython's C and Python ``Task`` both expose
    ``_fut_waiter`` (``None`` while the task is runnable); the suite checks it on the interpreter in
    use, and the project supports Python 3.11 to 3.14. Where it is absent
    (another loop implementation or a future CPython) the fallback is conservative: the task counts
    as runnable and a warning is logged once, so ``TestSession.settle`` times out loudly instead of
    returning while work may still be running.
    """
    global _warned_no_waiter  # noqa: PLW0603
    waiter = getattr(task, "_fut_waiter", _NO_WAITER)
    if waiter is _NO_WAITER:
        if not _warned_no_waiter:
            _warned_no_waiter = True
            logger.warning(
                "asyncio.Task has no _fut_waiter on this interpreter; settle() cannot tell an idle task "
                "from a runnable one and will wait for its timeout (the attribute is checked on Python 3.11 to 3.14)"
            )
        return False
    return waiter is not None and not waiter.done()


def _default_voice_factory(app: App) -> Any:
    """Build the voice agent of an app made by ``App.from_config`` (see :mod:`._voice`)."""
    from ._voice import build_voice_agent  # noqa: PLC0415 - keeps the app module free of the voice stack

    if not app._config.voice.enabled:
        return None
    return build_voice_agent(app, app._load_profile())


class _AppAgent(RuntimeAgent):
    """The app's own runtime subscriber: turns standard-topic events into scope operations."""

    def __init__(self, app: App) -> None:
        super().__init__()
        self._app = app

    @subscribe(PARTICIPANT_CONNECTED)
    async def connected(self, message: ParticipantConnected, _ctx: RuntimeContext) -> None:
        await self._app._contained(self._app._connected(message.connection), "connect")

    @subscribe(PARTICIPANT_DISCONNECTED)
    async def disconnected(self, message: ParticipantDisconnected, _ctx: RuntimeContext) -> None:
        await self._app._contained(self._app._disconnected(message.connection), "disconnect")

    @subscribe(USER_QUERY)
    async def query(self, message: UserQuery, ctx: RuntimeContext) -> None:
        if ctx.metadata.participant_id is not None:
            await self._app._contained(self._app._admit_turn(ctx.metadata.participant_id, message), "query")

    @subscribe(INTERRUPTED)
    async def interrupted(self, _message: Interrupted, ctx: RuntimeContext) -> None:
        await self._app._contained(self._app._interrupt(ctx.metadata.participant_id), "interrupt")


class _OutputBridge(RuntimeAgent):
    """Forward :data:`.topics.OUTPUT` to the voice agent's output topic.

    The original producer name travels with the message: the voice agent queues and withdraws
    updates per producer, which is what makes ``p.say(source=...)`` and ``p.withdraw`` per source.
    """

    def __init__(self, runtime: AgentRuntime) -> None:
        super().__init__()
        self._runtime = runtime

    @subscribe(OUTPUT)
    async def forward(self, message: ResponseOutput, ctx: RuntimeContext) -> None:
        from xr_ai_voice import VOICE_OUTPUT_TOPIC  # noqa: PLC0415

        await self._runtime.publish(
            VOICE_OUTPUT_TOPIC, message, participant_id=ctx.metadata.participant_id, source=ctx.metadata.source
        )


class _HubOutputBridge(RuntimeAgent):
    """Test-double output path: record :data:`.topics.OUTPUT` on a ``FakeHub``.

    A streamed reply is one spoken message, recorded when its final fragment arrives.
    """

    def __init__(self, hub: FakeHub) -> None:
        super().__init__()
        self._hub = hub
        self._partial: dict[tuple[str | None, str], list[str]] = {}

    @subscribe(OUTPUT)
    async def record(self, message: ResponseOutput, ctx: RuntimeContext) -> None:
        pid = ctx.metadata.participant_id
        if pid is None or message.kind == "withdraw":
            return
        if message.response_id is None:
            self._hub.publish_output(pid, message.text, urgent=message.urgent)
            return
        parts = self._partial.setdefault((pid, message.response_id), [])
        parts.append(message.text)
        if message.final:
            text = "".join(self._partial.pop((pid, message.response_id)))
            if text.strip():
                self._hub.publish_output(pid, text)


class App:
    """The app: owns every task, client and participant, so that app code only describes behavior.

    Build it with :meth:`from_config`, add components with :meth:`add`, answer turns with :meth:`on_turn`, and run
    it with :meth:`run` (or :meth:`start` and :meth:`stop` in your own event loop). The hub connection (presence and
    frames) is the ``Hub`` component, added on purpose.
    """

    def __init__(self, config: AppConfig, *, profile: str | None = None, enrollment: Enrollment | None = None) -> None:
        self._config = config
        self._enrollment = enrollment
        """The enrollment policy. ``None`` only for a hand-built ``App(config)``: connections are ignored."""
        self._warned_no_enrollment = False
        self._retention: dict[str, asyncio.Task[Any]] = {}
        """``Persistent`` enrollment: the app task that un-enrolls a participant ``keep_for_s`` after its last leave."""
        self._profile = profile  # explicit choice only; ``config.profile`` is the lowest-priority default
        self._runtime = AgentRuntime()
        self._components: list[Component] = []
        self._participants: dict[str, Participant] = {}
        self._leaving: dict[str, asyncio.Task[None]] = {}
        self._turn_handler: Callable[[Turn, Participant], Awaitable[Any]] | None = None
        self._closeables: list[Any] = []
        self._app_tasks: set[asyncio.Task[Any]] = set()
        self._abandoned: set[asyncio.Task[Any]] = set()
        self._component_scopes: dict[int, ParticipantScope] = {}
        """Component-lifetime work (``Component.spawn``), one scope per component, drained at stop."""
        self._app_state: ParticipantState | None = None
        self._tracer: _tracing.Tracer | None = None
        self._started = False
        self._admitting = False
        self._start_task: asyncio.Task[None] | None = None
        self._factory_loop: asyncio.AbstractEventLoop | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._trace: list[str] = []
        self._statuses: list[tuple[str, str, dict[str, Any]]] = []
        self._ready_file: Path | None = None
        self._probes: dict[str, _ready.Probe] = {}
        """Readiness: one probe per model client built (entry name to probe), checked before the app reports ready."""
        self._ready_wait: asyncio.Task[None] | None = None
        self._agent_status: Callable[[str, str], Awaitable[None]] | None = None
        """Where busy/idle goes on a device: the voice transport endpoint's ``set_status(status, participant_id)``."""
        self._runtime_registered = False
        # Replaceable for tests and for apps that supply their own voice stack: ``factory(app)``
        # returns an object with ``async run(runtime)``, or ``None`` to run without voice.
        self._voice_factory: Callable[[App], Any] | None = None
        self._voice: Any = None
        # Real bindings: the profile and the clients built from it, loaded on first use.
        self._profile_cache: Profile | None = None
        self._pool_cache: ClientPool | None = None
        self._roles: dict[str, Role] = {}
        # Test-double seam: set by ``_install_test_doubles``.
        self._hub: FakeHub | None = None
        self._fake_models: FakeModels | None = None
        self._admitted = asyncio.Event()
        """Set once ``start`` admits participants; the ``Hub`` component begins receiving connection events after it."""

    @classmethod
    def from_config(cls, path: str | Path, *, enrollment: Enrollment, profile: str | None = None) -> App:
        """Load ``path`` (YAML) and the model profile it selects; no network or process is started.

        ``enrollment`` is required and has no default. ``PerConnection(admit=...)`` enrolls an
        identity on its first connection and un-enrolls it when its last connection ends;
        ``Persistent(keep_for_s=..., admit=...)`` keeps the participant across disconnects. It is an app setting;
        components see only the hooks.

        The app built here runs against real services: ``app.role`` binds clients of the selected
        profile and ``start`` builds the voice agent (speech models, voice gate and hub transport)
        from the config and the profile. Nothing is connected before ``start``; clients are built
        when a role is first asked for and closed in the stop sequence. The hub connection is the
        ``Hub`` component; add it with :meth:`add`.
        """
        app = cls(_load_config(path), profile=profile, enrollment=enrollment)
        app._voice_factory = _default_voice_factory
        return app

    def settings(self, model: type[T]) -> T:
        """Return the ``settings:`` section of ``app.yaml`` validated as ``model`` (a pydantic model).

        Missing keys take the model's defaults. Raises ``ValueError`` listing every invalid or
        unknown-to-the-model field (pydantic's ``extra`` policy decides whether unknown keys fail).
        """
        try:
            return model.model_validate(dict(self._config.settings))
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(x) for x in e['loc']) or '<settings>'}: {e['msg']}" for e in exc.errors()
            )
            raise ValueError(f"invalid settings for {model.__name__}: {errors}") from exc

    def role(self, name: str) -> Role:
        """Return the model role ``name`` from the active profile."""
        if self._fake_models is not None:
            return self._fake_models.role(name)
        if (role := self._roles.get(name)) is not None:
            return role
        profile = self._load_profile()
        entry = profile.resolve(name)  # KeyError lists what the profile defines
        pool = self._pool(profile)
        category = _category(profile.models.entries[entry])
        client = pool.client(entry)
        role = Role.bind(name, entry=entry, category=category, client=client, limiter=pool.limiter(entry, name))
        if client is not None:
            self._probes.setdefault(entry, _ready.probe_for(category, client))
        self._roles[name] = role
        return role

    def _load_profile(self) -> Profile:
        """The profile from ``--profile``, then ``XR_AI_PROFILE``, then ``profile:`` in ``app.yaml``; loaded once."""
        if self._profile_cache is None:
            chosen = self._profile or os.environ.get(PROFILE_ENV) or self._config.profile
            self._profile_cache = load_profile(select_profile(self._config.config_dir, chosen))
        return self._profile_cache

    def _pool(self, profile: Profile) -> ClientPool:
        if self._pool_cache is None:
            self._pool_cache = ClientPool(profile, self._closeables.append)
        return self._pool_cache

    # ------------------------------------------------------------------ components and turns

    def add(self, component: C) -> C:
        """Add ``component`` to the app (before :meth:`start`) and return it; the order of adds fixes hook order."""
        if not isinstance(component, Component):
            raise TypeError("app.add takes a Component")
        if self._started or self._start_task is not None:
            raise RuntimeError("components must be added before app.start()")
        if component in self._components:
            raise ValueError("component is already added")
        component._app = self
        self._components.append(component)
        return component

    def on_turn(self, fn: F, /) -> F:
        """Make ``fn(turn, p)`` the one turn handler: traced, and seen first by turn observers (``Voice``)."""
        if self._turn_handler is not None:
            raise ValueError("an on_turn handler is already registered")
        self._turn_handler = fn
        return fn

    def on_start(self, fn: Callable[[], Awaitable[None]], /) -> Callable[[], Awaitable[None]]:
        """Run ``await fn()`` once the app has started (after the components' own ``on_start``)."""
        self.add(_StartHook(fn))
        return fn

    # ------------------------------------------------------------------ state and people

    @property
    def state(self) -> ParticipantState:
        """App-scoped state, shared by every participant, as long as the app runs."""
        if self._app_state is None:
            self._app_state = ParticipantState()
        return self._app_state

    @property
    def participants(self) -> list[Participant]:
        """Everyone enrolled now, in enrollment order."""
        return list(self._participants.values())

    # ------------------------------------------------------------------ trace

    @property
    def trace(self) -> _tracing.Tracer:
        """The app's trace: ``app.trace.add(sink)``, ``app.trace.emit(...)``, ``app.trace.span(...)``."""
        if self._tracer is None:
            self._tracer = _tracing.Tracer()
            _configure_trace(self._tracer, self._config.trace)
        return self._tracer

    # ------------------------------------------------------------------ start

    async def start(self) -> None:
        """Connect clients, warm up models, run ``on_start`` hooks and begin admitting participants."""
        _tracing.activate(self.trace)  # in the caller's context, which the app's tasks inherit
        if self._start_task is None:
            loop = asyncio.get_running_loop()
            install_relay_task_factory(loop)  # app code may use TaskGroup and gather (R1.9)
            self._factory_loop = loop
            self._start_task = loop.create_task(self._start())
        try:
            await asyncio.shield(self._start_task)
        except asyncio.CancelledError:
            if not asyncio.current_task().cancelling() and self._stop_task is not None:
                raise RuntimeError("app stopped during start") from None
            raise

    async def _start(self) -> None:
        if self._stop_task is not None:
            raise RuntimeError("app is stopped")
        self._trace.append("start")
        try:
            if not self._runtime_registered:
                self._runtime.register("xr-ai-app", _AppAgent(self))
                self._runtime_registered = True
            voice = self._voice_factory(self) if self._voice_factory is not None else None
            self._voice = voice
            if isinstance(voice, RuntimeAgent):
                # The voice agent receives spoken output through its own subscription to the voice output topic.
                self._runtime.register("xr-ai-voice", voice)
            if self._hub is not None:
                self._runtime.register("xr-ai-output-bridge", _HubOutputBridge(self._hub))
            elif voice is not None:
                self._runtime.register("xr-ai-output-bridge", _OutputBridge(self._runtime))
            await self._runtime.start()
            self._started = True
            for component in self._components:
                await self._run_hook(component.on_start(), "on_start", raise_errors=True)
            self._admitting = True
            self._admitted.set()
            if voice is not None:
                self._spawn_app_task(self._when_ready(voice.run(self._runtime)), "voice")
            elif self._ready_file is not None:
                self._spawn_app_task(self._when_ready(self._touch_ready_file()), "ready")
        except BaseException:
            logger.exception("app start failed; running the stop sequence")
            if self._stop_task is None:
                await asyncio.shield(self._ensure_stop(cancel_start=False))
            raise

    async def _when_ready(self, then: Coroutine[Any, Any, Any]) -> Any:
        """Wait until the model servers behind every bound role answer (see :mod:`._ready`), then run ``then``.

        ``then`` starts the voice pipeline (which reports ready once its transport is up) or writes the ready file.
        The stop sequence cancels the wait.
        """
        try:
            if self._probes:
                self._ready_wait = asyncio.get_running_loop().create_task(_ready.wait_until_ready(self._probes))
                await self._ready_wait
        except BaseException:
            then.close()
            raise
        return await then

    async def _touch_ready_file(self) -> None:
        assert self._ready_file is not None  # noqa: S101 - only scheduled with a ready file
        self._ready_file.parent.mkdir(parents=True, exist_ok=True)
        self._ready_file.touch()

    # ------------------------------------------------------------------- stop

    async def stop(self) -> None:
        """Run the stop sequence. Idempotent."""
        try:
            await asyncio.shield(self._ensure_stop())
        finally:
            self._release_task_factory()
            if self._tracer is not None:
                _tracing.deactivate(self._tracer)

    def _ensure_stop(self, *, cancel_start: bool = True) -> asyncio.Task[None]:
        if self._stop_task is None:
            self._stop_task = asyncio.get_running_loop().create_task(self._stop(cancel_start=cancel_start))
            # Every stop ends here, including the one a failed start runs without ``stop()`` being called.
            self._stop_task.add_done_callback(lambda _task: self._release_task_factory())
        return self._stop_task

    def _release_task_factory(self) -> None:
        if self._factory_loop is not None:
            uninstall_relay_task_factory(self._factory_loop)
            self._factory_loop = None

    async def _stop(self, *, cancel_start: bool = True) -> None:
        timeouts = self._config.timeouts
        trace = self._trace
        trace.append(STOP_SEQUENCE[0])
        self._admitting = False
        if cancel_start and self._start_task is not None and not self._start_task.done():
            self._start_task.cancel()
            await asyncio.gather(self._start_task, return_exceptions=True)
        if self._ready_wait is not None:
            self._ready_wait.cancel()  # still waiting for a model server: nothing to drain
        request_stop = getattr(self._voice, "request_stop", None)
        if request_stop is not None:
            request_stop()  # the voice session is stopped by the app, not by its own signal handlers (R7.3)
            await self._wait_app_tasks()
        await self._cancel_app_tasks()
        trace.append(STOP_SEQUENCE[1])
        pending = [self._begin_leave(pid) for pid in tuple(self._participants)]
        pending += [task for task in self._leaving.values() if task not in pending]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._component_scopes:  # component-lifetime work ends after every participant left, before the stop hooks
            await asyncio.gather(*(s.leave() for s in self._component_scopes.values()), return_exceptions=True)
        trace.append(STOP_SEQUENCE[2])
        if self._started:
            for component in reversed(self._components):
                await self._run_hook(component.on_stop(), "on_stop")
        trace.append(STOP_SEQUENCE[3])
        for client in reversed(self._closeables):
            await self._bounded(self._close_client(client), timeouts.close_s, f"close {client!r}")
        trace.append(STOP_SEQUENCE[4])
        if self._started:
            try:
                await self._runtime.stop()
            except Exception:  # noqa: BLE001 - containment boundary (rule R4)
                logger.exception("runtime stop failed")

    @staticmethod
    async def _close_client(client: Any) -> None:
        for name in ("aclose", "close"):
            method = getattr(client, name, None)
            if method is not None:
                result = method()
                if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                    await result
                return

    # ----------------------------------------------------------- task helpers

    def _spawn_app_task(self, coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
        """Start an app-level task (rule R1.6); it is cancelled and drained in the stop sequence."""
        if not self._admitting:
            coro.close()
            raise RuntimeError("app is not admitting tasks")
        task = asyncio.get_running_loop().create_task(self._contained(coro, name), name=f"app:{name}")
        self._app_tasks.add(task)
        task.add_done_callback(self._app_tasks.discard)
        return task

    def _spawn_component_task(
        self, component: Component, coro: Coroutine[Any, Any, Any], *, key: str | None, supersede: bool
    ) -> asyncio.Task[Any]:
        """Run ``coro`` for ``component`` (``Component.spawn``); the stop sequence cancels and drains it."""
        if self._stop_task is not None:
            coro.close()
            raise RuntimeError("app is stopping")
        scope = self._component_scopes.get(id(component))
        if scope is None:
            scope = ParticipantScope(f"component:{type(component).__name__}", timeouts=self._config.timeouts)
            scope._abandoned = self._abandoned
            self._component_scopes[id(component)] = scope
        return scope.spawn(coro, key=key, supersede=supersede)

    async def _wait_app_tasks(self) -> None:
        tasks = {t for t in self._app_tasks if t is not asyncio.current_task() and not t.done()}
        if tasks:
            await asyncio.wait(tasks, timeout=self._config.timeouts.drain_s)

    async def _cancel_app_tasks(self) -> None:
        tasks = {t for t in self._app_tasks if t is not asyncio.current_task() and not t.done()}
        for task in tasks:
            task.cancel()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=self._config.timeouts.drain_s)
            for task in pending:
                logger.error("app task {} ignored cancellation; abandoned", task.get_name())
                self._abandoned.add(task)

    async def _contained(self, coro: Coroutine[Any, Any, Any], label: str) -> Any:
        """Await ``coro`` and log, rather than propagate, any exception (rule R4)."""
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - containment boundary (rule R4)
            logger.exception("app {} failed", label)
            return None

    async def _bounded(self, coro: Coroutine[Any, Any, Any], timeout_s: float, label: str) -> bool:
        """Run ``coro`` for at most ``timeout_s``; abandon it (even if it ignores cancellation) after."""
        task = asyncio.get_running_loop().create_task(coro, name=label)
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
        if not done:
            task.cancel()
            self._abandoned.add(task)
            task.add_done_callback(self._abandoned.discard)
            logger.error("{} exceeded {}s and was abandoned", label, timeout_s)
            return False
        if not task.cancelled() and task.exception() is not None:
            logger.opt(exception=task.exception()).error("{} failed", label)
            return False
        return True

    async def _run_hook(self, coro: Coroutine[Any, Any, Any], label: str, *, raise_errors: bool = False) -> bool:
        """Run one component hook bounded by ``hook_s``; log failures and keep going (rule R2.7)."""
        if raise_errors:
            return bool(await self._contained_raise(coro, label))
        return await self._bounded(coro, self._config.timeouts.hook_s, label)

    async def _contained_raise(self, coro: Coroutine[Any, Any, Any], label: str) -> bool:
        # on_start: a failure aborts startup (R3.3); a hang is bounded by ``start_s`` (default ``hook_s``).
        timeouts = self._config.timeouts
        bound = timeouts.hook_s if timeouts.start_s is None else timeouts.start_s
        task = asyncio.get_running_loop().create_task(coro, name=label)
        try:
            done, _ = await asyncio.wait({task}, timeout=bound)
        except asyncio.CancelledError:
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=timeouts.hook_s)
            if not done:
                self._abandoned.add(task)
                task.add_done_callback(self._abandoned.discard)
            raise
        if not done:
            task.cancel()
            self._abandoned.add(task)
            task.add_done_callback(self._abandoned.discard)
            raise TimeoutError(f"{label} exceeded {bound}s")
        task.result()
        return True

    # ----------------------------------------------------------- participants

    async def _set_agent_status(self, p: Participant, status: str) -> None:
        """Tell the device whether the agent is answering ``p`` (``processing``) or not (``idle``); traced.

        It goes through the voice transport's endpoint, the one that announces readiness (the client folds every
        agent's status into one), or to a test session's hub. Never raises.
        """
        self.trace.emit("status", status)
        try:
            if self._hub is not None:
                self._hub.publish_agent_status(p.id, status)
            elif self._agent_status is not None:
                await self._agent_status(status, p.id)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - containment boundary (rule R4): a lost status is not a failed turn
            logger.opt(exception=True).debug("agent status {} for {} not sent", status, p.id)

    async def _publish_status(self, participant_id: str, text: str, detail: dict[str, Any]) -> None:
        """Report a participant's status (a failed task's ``error``): logged, and recorded by a test session's hub."""
        self._statuses.append((participant_id, text, detail))
        logger.info("status {}: {} {}", participant_id, text, detail)
        if self._hub is not None:
            try:
                self._hub.publish_status(participant_id, text, **detail)
            except RuntimeClosedError:
                logger.debug("status for {} dropped: hub closed", participant_id)

    async def _connected(self, connection: Connection) -> None:
        """A device connected (``PARTICIPANT_CONNECTED``): enroll the identity or add a connection."""
        participant_id = connection.identity
        enrollment = self._enrollment
        if enrollment is None:
            if not self._warned_no_enrollment:
                self._warned_no_enrollment = True
                logger.warning(
                    "connection of {} ignored: the app has no enrollment policy (App.from_config(enrollment=...))",
                    participant_id,
                )
            return
        if not self._admitting:
            logger.warning("connection of {} arrived while the app is not admitting; dropped", participant_id)
            return
        admit = enrollment.admit
        if admit is not None and not admit(connection):
            logger.info("connection of {} rejected by admit", participant_id)
            return
        leaving = self._leaving.get(participant_id)
        if leaving is not None:
            await asyncio.shield(leaving)  # an un-enroll in progress finishes first; the identity enrolls afresh (R2.4)
        p = self._participants.get(participant_id)
        if p is None:
            await self._enroll(participant_id, connection)
            return
        self._cancel_retention(participant_id)
        p._add_connection(connection)

    async def _disconnected(self, connection: Connection) -> None:
        """A device disconnected (``PARTICIPANT_DISCONNECTED``): drop it; the policy decides when to un-enroll."""
        participant_id = connection.identity
        p = self._participants.get(participant_id)
        if p is None or not p._remove_connection(connection):
            return  # not enrolled, or a connection that was rejected (or is already gone)
        enrollment = self._enrollment
        if p.connected or enrollment is None or not self._admitting:
            return  # another device is still connected, or the stop sequence un-enrolls everyone
        keep_for_s = enrollment.keep_for_s
        if keep_for_s is None:
            return
        if keep_for_s <= 0:
            await self._leave(participant_id)
            return
        self._cancel_retention(participant_id)
        try:
            self._retention[participant_id] = self._spawn_app_task(
                self._retain(participant_id, p, keep_for_s), f"retain:{participant_id}"
            )
        except RuntimeError:
            logger.debug("retention of {} not started: the app is stopping", participant_id)

    async def _retain(self, participant_id: str, p: Participant, keep_for_s: float) -> None:
        """``Persistent`` enrollment: un-enroll ``p`` once ``keep_for_s`` passed with no connection (an app task)."""
        await asyncio.sleep(keep_for_s)
        if self._retention.get(participant_id) is asyncio.current_task():
            del self._retention[participant_id]
        if self._participants.get(participant_id) is p and not p.connected:
            await self._leave(participant_id)

    def _cancel_retention(self, participant_id: str) -> None:
        task = self._retention.pop(participant_id, None)
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _enroll(self, participant_id: str, connection: Connection) -> None:
        if not self._admitting:
            logger.warning("participant {} enrolled while the app is not admitting; dropped", participant_id)
            return
        p = Participant(participant_id)
        scope = ParticipantScope(participant_id, timeouts=self._config.timeouts)
        p._bind(
            scope=scope,
            runtime=self._runtime,
            artifacts=(
                ArtifactSink(self._config.artifacts_dir, participant_id=participant_id)
                if self._config.artifacts_dir is not None
                else None
            ),
        )
        scope._abandoned = self._abandoned  # one set, so run() can see every task that ignored cancellation
        scope._bind(
            on_error=lambda key, exc: self._publish_status(
                participant_id, "error", {"task": key, "error": f"{type(exc).__name__}: {exc}"}
            ),
            run_hooks=lambda: self._run_unenrolled_hooks(p),
            release=p._release,
        )
        p._add_connection(connection)
        self._participants[participant_id] = p
        for component in self._components:
            # Bounded: this runs inside the delivery the publisher (the hub component) awaits.
            await self._run_hook(component.on_participant_enrolled(p), f"on_participant_enrolled {participant_id}")

    async def _run_unenrolled_hooks(self, p: Participant) -> None:
        for component in reversed(self._components):
            await self._run_hook(component.on_participant_unenrolled(p), f"on_participant_unenrolled {p.id}")

    def _begin_leave(self, participant_id: str) -> asyncio.Task[None]:
        task = self._leaving.get(participant_id)
        if task is not None:
            return task
        p = self._participants.get(participant_id)
        if p is None:
            done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            done.set_result(None)
            return done  # type: ignore[return-value]
        self._cancel_retention(participant_id)

        async def run() -> None:
            try:
                await p._scope.leave()
            finally:
                if self._participants.get(participant_id) is p:
                    del self._participants[participant_id]

        task = asyncio.get_running_loop().create_task(run(), name=f"app:unenroll:{participant_id}")
        self._leaving[participant_id] = task
        task.add_done_callback(lambda t: self._forget_leave(participant_id, t))
        return task

    def _forget_leave(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._leaving.get(participant_id) is task:
            del self._leaving[participant_id]

    async def _leave(self, participant_id: str) -> None:
        """Run the un-enroll sequence (R2); its own task, so a cancelled delivery cannot cut it short."""
        await asyncio.shield(self._begin_leave(participant_id))

    async def _admit_turn(self, participant_id: str, query: UserQuery) -> None:
        p = self._participants.get(participant_id)
        handler = self._turn_handler
        if p is None or not p._scope.admitting or not self._admitting:
            logger.debug("query for {} dropped: not admitting", participant_id)
            return
        if handler is None:
            logger.debug("query for {} dropped: no on_turn handler", participant_id)
            return
        turn = Turn(text=query.text, timestamp_us=query.timestamp_us, id=uuid.uuid4().hex)
        p.spawn(self._run_turn(handler, turn, p), key="turn", supersede=True)

    async def _run_turn(
        self, handler: Callable[[Turn, Participant], Awaitable[Any]], turn: Turn, p: Participant
    ) -> None:
        """One turn, in the participant's ``turn`` task: traced, and seen first by every ``_observe_turn`` component.

        ``Voice`` observes turns to record the user's words in its dialogue before the handler answers. The device
        shows the agent as ``processing`` while the turn runs and ``idle`` after (a superseding turn starts only once
        the one it replaces has ended, so the two never overlap).
        """
        CURRENT_TURN.set(turn)
        with _tracing.bound(participant=p.id, session=p.session.id, turn=turn.id, task=None, parent=None):
            started = time.monotonic()
            self.trace.emit("turn", "", phase="start", id=turn.id, text=turn.text, timestamp_us=turn.timestamp_us)
            outcome = "done"
            await self._set_agent_status(p, "processing")
            try:
                for component in list(self._components):
                    observe = getattr(component, "_observe_turn", None)
                    if observe is not None:
                        await observe(turn, p)
                await handler(turn, p)
            except BaseException as exc:
                outcome = type(exc).__name__
                raise
            finally:
                await self._set_agent_status(p, "idle")
                self.trace.emit(
                    "turn", "", phase="end", id=turn.id, outcome=outcome, duration_s=time.monotonic() - started
                )

    async def _interrupt(self, participant_id: str | None) -> None:
        targets = (
            [self._participants[participant_id]]
            if participant_id in self._participants
            else (list(self._participants.values()) if participant_id is None else [])
        )
        for p in targets:
            await p.cancel("turn")

    # -------------------------------------------------------------------- run

    def run(self, argv: Sequence[str] | None = None) -> None:
        """Parse ``--config``, ``--profile`` and ``--ready-file``, install signal handlers, run until stopped.

        The only place in the SDK that installs signal handlers.
        """
        parser = argparse.ArgumentParser(description="Run an xr_ai_app app")
        parser.add_argument("--config", type=Path, help="app.yaml; replaces the config this app was built with")
        parser.add_argument("--profile", help="model profile name or path")
        parser.add_argument("--ready-file", type=Path, help="file created once the app is ready")
        args = parser.parse_args(argv)
        if args.config is not None:
            self._config = _load_config(args.config)
        if args.profile is not None:
            self._profile = args.profile
        if args.config is not None or args.profile is not None:
            self._profile_cache = self._pool_cache = None  # clients are built lazily, so nothing is open yet
            self._roles.clear()
        self._ready_file = args.ready_file
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._run_until_stopped())
        except asyncio.CancelledError:
            raise SystemExit(130) from None
        finally:
            self._close_loop(loop)

    def _close_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Close ``loop``; ``asyncio.run`` would wait forever for a task that ignores cancellation."""
        try:
            if not any(not t.done() for t in self._abandoned):
                leftovers = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for task in leftovers:
                    task.cancel()
                if leftovers:
                    loop.run_until_complete(asyncio.gather(*leftovers, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    async def _run_until_stopped(self) -> None:
        loop = asyncio.get_running_loop()
        stop_requested = asyncio.Event()
        main = asyncio.current_task()
        signals_seen = 0

        def on_signal() -> None:
            nonlocal signals_seen
            signals_seen += 1
            if signals_seen == 1:
                stop_requested.set()
            elif main is not None:
                main.cancel()  # a second signal abandons the wait; every stop step is already bounded

        installed: list[signal.Signals] = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, on_signal)
                installed.append(sig)
        try:
            await self.start()
            voice_done = [t for t in self._app_tasks if t.get_name() == "app:voice"]
            waiters = [asyncio.ensure_future(stop_requested.wait()), *voice_done]
            try:
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiters[0].cancel()
        finally:
            try:
                await self.stop()
            finally:
                for sig in installed:
                    loop.remove_signal_handler(sig)

    def test_session(
        self, *, models: FakeModels | None = None, replace: Mapping[type, Component] | None = None
    ) -> AbstractAsyncContextManager[TestSession]:
        """Start the app on a fake hub and scripted ``models``; the session's ``trace`` holds every event.

        The session connects, speaks, shows frames and leaves; ``replace`` swaps an added component by class.
        """
        from .testing import test_session  # noqa: PLC0415 - the test kit imports this module

        return test_session(self, models=models, replace=replace)

    # ------------------------------------------------ test-double seam

    def _install_test_doubles(
        self, hub: FakeHub, models: FakeModels, *, replace: Mapping[type, Component] | None = None
    ) -> None:
        """Replace the hub connection with ``hub`` and resolve roles from ``models``; call before :meth:`start`.

        ``hub`` (a ``FakeHub``, which is a ``Hub`` replacement) takes the place of the added ``Hub`` component, or is
        added last when the app has none, so a test session always has a device side. ``replace`` swaps any other
        added component by class (the replacement takes the old one's position). Test-kit convenience, not an SDK
        default: an app with no enrollment policy gets ``PerConnection(admit=None)`` here. Builds no real client or
        process. Output and status go to the hub's record.
        """
        from .hub import Hub  # noqa: PLC0415 - keeps the app module free of the hub client

        if self._started or self._start_task is not None:
            raise RuntimeError("test doubles must be installed before app.start()")
        self._hub = hub
        self._fake_models = models
        # Roles bound before this call (a sample's module-level ``app.role(...)``) become fakes in place, so a test
        # session never reaches a real model endpoint.
        for name, role in list(self._roles.items()):
            fake = models.role(name)
            role.__class__ = type(fake)
            role.__dict__.clear()
            role.__dict__.update(fake.__dict__)
        self._voice_factory = None
        if self._enrollment is None:
            self._enrollment = PerConnection(admit=None)
        self._swap_component(Hub, hub, add_if_missing=True)
        for cls, new in (replace or {}).items():
            self._swap_component(cls, new, add_if_missing=False)

    def _swap_component(self, cls: type, new: Component, *, add_if_missing: bool) -> None:
        """Replace every added instance of ``cls`` with ``new`` (before start); add ``new`` last when allowed."""
        positions = [i for i, c in enumerate(self._components) if isinstance(c, cls)]
        if not positions and not add_if_missing:
            raise ValueError(f"replace[{cls.__name__}]: the app has no {cls.__name__} to replace")
        new._app = self
        if not positions:
            self._components.append(new)
            return
        self._components[positions[0]] = new
        self._components = [c for i, c in enumerate(self._components) if i not in positions[1:]]

    def _is_idle(self) -> bool:
        """Whether no scope has a runnable task and no leave is running (used by ``TestSession.settle``).

        A task is runnable when it is not parked on an unfinished future. A task blocked on a sleep,
        an event or a model call is idle, but a pending ``p.artifacts`` write is not, because the task
        waiting on its thread would look parked.
        Tasks spawned under a key other than ``"turn"`` are long-lived
        by convention (watchers, procedures) and are ignored, because a busy loop would otherwise keep
        ``settle`` from ever returning; in-flight fake model calls are tracked by ``FakeModels``
        itself. This reads the private ``_fut_waiter`` of ``asyncio.Task``; see
        :func:`_parked` for the tested Python versions and the fallback when it is missing.
        """
        if self._leaving:
            return False
        for p in self._participants.values():
            if p._artifacts is not None and p._artifacts.pending:
                return False  # a task parked on a sink thread write is not idle
            long_lived = {t for key, t in p._scope._keyed.items() if key != "turn"}
            for task in p._scope._tasks:
                if task.done() or task in long_lived:
                    continue
                if not _parked(task):
                    return False
        return True


class _StartHook(Component):
    """Runs one ``app.on_start`` function when the app starts."""

    def __init__(self, fn: Callable[[], Awaitable[None]]) -> None:
        self._fn = fn

    async def on_start(self) -> None:
        await self._fn()


def _configure_trace(tracer: _tracing.Tracer, config: Mapping[str, Any]) -> None:
    """Add the sinks ``app.yaml``'s ``trace:`` asks for."""
    if config.get("jsonl") is not None:
        tracer.add(_tracing.JsonlSink(config["jsonl"]))
    if config.get("log"):
        level = config["log"] if isinstance(config["log"], str) else "DEBUG"
        tracer.add(_tracing.LogSink(level))
    if config.get("relay"):
        from ._relay import RelaySink  # noqa: PLC0415 - Relay is only needed when asked for

        tracer.add(RelaySink())


__all__ = ["App", "AppConfig", "VoiceConfig"]
