# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""App lifecycle on fakes: hooks, leave and stop sequences, containment, config and run (rules R1 to R7)."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import textwrap
from pathlib import Path

import pytest
from xr_ai_app import Component
from xr_ai_app._base.app import App, AppConfig
from xr_ai_app._base.enrollment import Connection, PerConnection
from xr_ai_app._base.participant import Participant, Turn
from xr_ai_app._base.scope import STOP_SEQUENCE, UNENROLL_SEQUENCE, ScopeTimeouts
from xr_ai_app._base.topics import (
    INTERRUPTED,
    OUTPUT,
    PARTICIPANT_CONNECTED,
    PARTICIPANT_DISCONNECTED,
    USER_QUERY,
    ParticipantConnected,
    ParticipantDisconnected,
)
from xr_ai_interaction import Interrupted, ResponseOutput, UserQuery
from xr_ai_runtime import Agent as RuntimeAgent
from xr_ai_runtime import RuntimeContext, subscribe


def make_app(tmp_path: Path | None = None, **timeouts: float) -> App:
    config = AppConfig(
        config_dir=Path("."),
        artifacts_dir=tmp_path,
        timeouts=ScopeTimeouts(**{"drain_s": 0.2, "hook_s": 0.2, "close_s": 0.2, **timeouts}),
    )
    return App(config, enrollment=PerConnection(admit=None))


async def join(app: App, pid: str = "p1") -> None:
    connection = Connection(pid, "s1")
    await app._runtime.publish(PARTICIPANT_CONNECTED, ParticipantConnected(connection=connection), participant_id=pid)


async def leave(app: App, pid: str = "p1") -> None:
    connection = Connection(pid, "s1")
    await app._runtime.publish(
        PARTICIPANT_DISCONNECTED, ParticipantDisconnected(connection=connection), participant_id=pid
    )


async def ask(app: App, text: str, pid: str = "p1") -> None:
    await app._runtime.publish(USER_QUERY, UserQuery(text=text, timestamp_us=1), participant_id=pid)


class Recorder(Component):
    """Records hook order and the output it sees."""

    def __init__(self, name: str, log: list[str]) -> None:
        super().__init__()
        self.name = name
        self.log = log

    async def on_start(self) -> None:
        self.log.append(f"{self.name}:start")

    async def on_participant_enrolled(self, p: Participant) -> None:
        self.log.append(f"{self.name}:joined:{p.id}")

    async def on_participant_unenrolled(self, p: Participant) -> None:
        self.log.append(f"{self.name}:left:{p.id}:state={dict(p.state)}")

    async def on_stop(self) -> None:
        self.log.append(f"{self.name}:stop")


class Spoken(RuntimeAgent):
    """What the app publishes as output, heard on its runtime (registered before ``start``, see :func:`listen`)."""

    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[tuple[str | None, ResponseOutput]] = []

    @subscribe(OUTPUT)
    async def collect(self, message: ResponseOutput, ctx: RuntimeContext) -> None:
        self.outputs.append((ctx.metadata.participant_id, message))


def listen(app: App) -> Spoken:
    return app._runtime.register("spoken", Spoken())


def user_tasks() -> set[asyncio.Task[object]]:
    me = asyncio.current_task()
    return {t for t in asyncio.all_tasks() if t is not me and not t.done()}


async def test_hooks_run_in_registration_order_on_start_and_join_and_reversed_on_leave_and_stop() -> None:
    app = make_app()
    log: list[str] = []
    app.add(Recorder("a", log))
    app.add(Recorder("b", log))
    await app.start()
    await join(app)
    await leave(app)
    await app.stop()
    assert log == [
        "a:start", "b:start",
        "a:joined:p1", "b:joined:p1",
        "b:left:p1:state={}", "a:left:p1:state={}",
        "b:stop", "a:stop",
    ]  # fmt: skip


async def test_hooks_see_state_intact_on_leave_and_it_is_released_after() -> None:
    app = make_app()
    log: list[str] = []

    class Setter(Component):
        async def on_participant_enrolled(self, p: Participant) -> None:
            p.state.step = 3

    app.add(Setter())
    app.add(Recorder("r", log))
    await app.start()
    await join(app)
    p = app._participants["p1"]
    await leave(app)
    assert log[-1] == "r:left:p1:state={'step': 3}"
    assert dict(p.state) == {}
    await app.stop()


async def test_stop_runs_the_five_steps_in_order_and_leaves_no_task_alive() -> None:
    before = user_tasks()
    app = make_app()
    app.add(Recorder("a", []))
    closed: list[str] = []

    class Client:
        async def aclose(self) -> None:
            closed.append("client")

    app._closeables.append(Client())
    await app.start()
    for pid in ("p1", "p2"):
        await join(app, pid)
        for key in ("turn", "loop", "x"):
            app._participants[pid].spawn(asyncio.sleep(10), key=key)
    await app.stop()
    assert app._trace == ["start", *STOP_SEQUENCE]
    assert closed == ["client"]
    assert app._runtime.running is False
    await asyncio.sleep(0)
    assert user_tasks() == before


async def test_leave_runs_the_five_steps_in_order() -> None:
    app = make_app()
    await app.start()
    await join(app)
    scope = app._participants["p1"]._scope
    await leave(app)
    assert scope._trace == list(UNENROLL_SEQUENCE)
    assert "p1" not in app._participants
    await app.stop()


async def test_stop_is_idempotent_and_a_second_caller_waits_for_the_first() -> None:
    app = make_app()
    log: list[str] = []

    class Slow(Component):
        async def on_stop(self) -> None:
            log.append("stop:begin")
            await asyncio.sleep(0.05)
            log.append("stop:end")

    app.add(Slow())
    await app.start()
    await asyncio.gather(app.stop(), app.stop())
    await app.stop()
    assert log == ["stop:begin", "stop:end"]


async def test_one_participants_failing_task_does_not_affect_another() -> None:
    app = make_app()
    await app.start()
    await join(app, "a")
    await join(app, "b")
    done = asyncio.Event()

    async def boom() -> None:
        raise RuntimeError("bad")

    async def fine() -> None:
        await asyncio.sleep(0.02)
        done.set()

    app._participants["b"].spawn(fine(), key="work")
    await asyncio.wait({app._participants["a"].spawn(boom(), key="boom")})
    await asyncio.wait_for(done.wait(), 1)
    assert app._participants["a"]._scope.admitting and app._participants["b"]._scope.admitting
    assert [(pid, text, d["task"]) for pid, text, d in app._statuses] == [("a", "error", "boom")]
    await app.stop()


async def test_leaving_one_participant_does_not_cancel_or_delay_another() -> None:
    app = make_app()
    await app.start()
    await join(app, "a")
    await join(app, "b")
    task_b = app._participants["b"].spawn(asyncio.sleep(10), key="work")
    await leave(app, "a")
    await asyncio.sleep(0)
    assert not task_b.done()
    await app.stop()
    assert task_b.cancelled()


async def test_a_superseded_turn_is_cancelled_before_the_next_one_starts() -> None:
    app = make_app()
    log: list[str] = []
    gate = asyncio.Event()

    @app.on_turn
    async def handle(turn: Turn, p: Participant) -> None:
        log.append(f"start:{turn.text}")
        try:
            if turn.text == "one":
                gate.set()
                await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0.01)
            log.append(f"end:{turn.text}")

    await app.start()
    await join(app)
    await ask(app, "one")
    await asyncio.wait_for(gate.wait(), 1)
    await ask(app, "two")
    await app._participants["p1"]._scope._keyed["turn"]
    assert log == ["start:one", "end:one", "start:two", "end:two"]
    await app.stop()


async def test_interrupted_cancels_the_current_turn_only() -> None:
    app = make_app()
    started = asyncio.Event()

    @app.on_turn
    async def handle(turn: Turn, p: Participant) -> None:
        started.set()
        await asyncio.sleep(10)

    await app.start()
    await join(app, "a")
    await join(app, "b")
    other = app._participants["b"].spawn(asyncio.sleep(10), key="other")
    await ask(app, "go", "a")
    await asyncio.wait_for(started.wait(), 1)
    turn = app._participants["a"]._scope._keyed["turn"]
    await app._runtime.publish(INTERRUPTED, Interrupted(), participant_id="a")
    assert turn.cancelled() and not other.done()
    await app.stop()


async def test_a_query_after_leave_is_dropped_and_a_turn_reply_is_a_reply_to_it() -> None:
    app = make_app()
    spoken = listen(app)

    @app.on_turn
    async def handle(turn: Turn, p: Participant) -> None:
        await p.reply(f"echo {turn.text}")

    await app.start()
    await join(app)
    await ask(app, "hi")
    await app._participants["p1"]._scope._keyed["turn"]
    replies = [(pid, out) for pid, out in spoken.outputs if out.kind == "reply"]
    assert {pid for pid, _ in replies} == {"p1"}
    assert "".join(out.text for _, out in replies) == "echo hi"
    assert len({out.request_id for _, out in replies}) == 1 and replies[-1][1].final
    await leave(app)
    await ask(app, "again")  # no participant: dropped, nothing raised
    await app.stop()


async def test_a_hook_that_raises_does_not_stop_the_remaining_hooks() -> None:
    app = make_app()
    log: list[str] = []

    class Bad(Component):
        async def on_participant_unenrolled(self, p: Participant) -> None:
            raise RuntimeError("hook failed")

        async def on_participant_enrolled(self, p: Participant) -> None:
            raise RuntimeError("join hook failed")

    app.add(Recorder("first", log))
    app.add(Bad())
    await app.start()
    await join(app)
    await leave(app)
    assert "first:left:p1:state={}" in log and "first:joined:p1" in log
    await app.stop()


async def test_a_hook_that_hangs_is_abandoned_after_its_timeout_and_logged() -> None:
    app = make_app(hook_s=0.05)
    log: list[str] = []
    release = asyncio.Event()

    class Hangs(Component):
        async def on_participant_unenrolled(self, p: Participant) -> None:
            while not release.is_set():
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    continue  # ignores cancellation

        async def on_stop(self) -> None:
            await asyncio.sleep(10)

    app.add(Recorder("first", log))
    app.add(Hangs())
    messages: list[str] = []
    from loguru import logger

    sink = logger.add(messages.append, level="ERROR")
    try:
        await app.start()
        await join(app)
        await asyncio.wait_for(leave(app), 2)  # Hangs runs first (reversed) and must not block "first"
        assert "first:left:p1:state={}" in log
        await asyncio.wait_for(app.stop(), 2)
    finally:
        logger.remove(sink)
        release.set()
        for task in tuple(app._abandoned):
            task.cancel()
    text = "".join(messages)
    assert "on_participant_unenrolled p1 exceeded" in text and "on_stop exceeded" in text


async def test_on_start_has_its_own_bound_that_defaults_to_hook_s() -> None:
    class Slow(Component):
        async def on_start(self) -> None:
            await asyncio.sleep(0.2)

    app = make_app(hook_s=0.05)  # default start_s: on_start is bounded by hook_s
    app.add(Slow())
    with pytest.raises(TimeoutError, match="exceeded 0.05s"):
        await app.start()
    app = make_app(hook_s=0.05, start_s=2.0)  # a long warm-up does not need a wide hook_s
    app.add(Slow())
    await app.start()
    await app.stop()


def test_app_yaml_timeouts_accepts_start_s(tmp_path: Path) -> None:
    path = write_config(tmp_path, "timeouts: {hook_s: 2, start_s: 90}\n")
    app = App.from_config(path, enrollment=PerConnection(admit=None))
    assert app._config.timeouts == ScopeTimeouts(hook_s=2.0, start_s=90.0)


def test_app_yaml_start_s_null_means_hook_s_and_bad_values_are_named(tmp_path: Path) -> None:
    path = write_config(tmp_path, "timeouts: {hook_s: 2, start_s: null}\n")
    assert App.from_config(path, enrollment=PerConnection(admit=None))._config.timeouts == ScopeTimeouts(hook_s=2.0)
    bad = tmp_path / "bad.yaml"
    for body, match in (
        ("timeouts: {hook_s: null}", "timeouts.hook_s must be a number"),
        ("timeouts: {start_s: soon}", "timeouts.start_s must be a number"),
        ("timeouts: {boot_s: 1}", "accepts drain_s, hook_s, close_s and start_s"),
    ):
        bad.write_text(body)
        with pytest.raises(ValueError, match=match):
            App.from_config(bad, enrollment=PerConnection(admit=None))


async def test_a_join_hook_that_hangs_is_bounded_so_the_publisher_is_not_blocked() -> None:
    app = make_app(hook_s=0.05)

    class Hangs(Component):
        async def on_participant_enrolled(self, p: Participant) -> None:
            await asyncio.sleep(10)

    app.add(Hangs())
    await app.start()
    await asyncio.wait_for(join(app), 1)  # publish returns; the hook was abandoned
    assert "p1" in app._participants
    await app.stop()


async def test_on_start_failure_runs_the_stop_sequence_for_what_started_and_reraises() -> None:
    app = make_app()
    log: list[str] = []

    class Fails(Component):
        async def on_start(self) -> None:
            raise RuntimeError("no model")

    app.add(Recorder("a", log))
    app.add(Fails())
    with pytest.raises(RuntimeError, match="no model"):
        await app.start()
    assert log == ["a:start", "a:stop"]
    assert app._trace[-1] == STOP_SEQUENCE[-1]
    assert app._runtime.running is False


async def test_participants_are_not_admitted_before_on_start_finishes() -> None:
    app = make_app()
    log: list[str] = []

    class Early(Component):
        async def on_start(self) -> None:
            await join(self.app)  # arrives during on_start: must be dropped
            log.append("on_start done")

    app.add(Early())
    await app.start()
    assert app._participants == {}
    await app.stop()


async def test_reenrolling_the_same_id_gets_fresh_state_scope_and_no_stale_tasks() -> None:
    app = make_app()
    await app.start()
    await join(app)
    first = app._participants["p1"]
    first.state.n = 1
    stale = first.spawn(asyncio.sleep(10), key="k")
    await leave(app)  # PerConnection: the last disconnect un-enrolls
    await join(app)
    second = app._participants["p1"]
    assert second is not first and "n" not in second.state
    assert stale.cancelled() and second._scope is not first._scope
    await app.stop()


async def test_nothing_publishes_into_the_runtime_after_stop_and_nothing_raises() -> None:
    app = make_app()
    listen(app)
    await app.start()
    await join(app)
    p = app._participants["p1"]
    await app.stop()
    await p.say("late")
    await p.reply("late")


async def test_add_is_rejected_after_start_and_for_non_components() -> None:
    app = make_app()
    with pytest.raises(TypeError):
        app.add(object())  # type: ignore[arg-type]
    agent = app.add(Recorder("a", []))
    assert agent.app is app
    with pytest.raises(ValueError, match="already"):
        app.add(agent)
    await app.start()
    with pytest.raises(RuntimeError, match="before app.start"):
        app.add(Recorder("b", []))
    await app.stop()


def test_a_component_with_a_subscribe_handler_is_rejected_and_told_the_alternative() -> None:
    # A component is not a runtime agent: a @subscribe method would never be called, so defining one fails.
    with pytest.raises(TypeError, match=r"Listens\.collect is a foundation @subscribe handler.*app\.on_turn"):

        class Listens(Component):
            @subscribe(OUTPUT)
            async def collect(self, message: ResponseOutput, ctx: RuntimeContext) -> None: ...

    class Mixin:
        @subscribe(OUTPUT)
        async def collect(self, message: ResponseOutput, ctx: RuntimeContext) -> None: ...

    with pytest.raises(TypeError, match=r"FromMixin\.collect .*p\.connected"):

        class FromMixin(Mixin, Component):
            pass


def test_an_unregistered_agent_explains_itself() -> None:
    with pytest.raises(RuntimeError, match=r"app\.add\(component\)"):
        Recorder("x", []).app  # noqa: B018


async def test_a_second_on_turn_handler_is_rejected() -> None:
    app = make_app()

    async def one(turn: Turn, p: Participant) -> None: ...

    assert app.on_turn(one) is one
    with pytest.raises(ValueError, match="already"):
        app.on_turn(one)


class FakeVoice:
    def __init__(self) -> None:
        self.cancelled = False
        self.running = asyncio.Event()

    async def run(self, runtime: object) -> None:
        self.running.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def test_voice_runs_as_an_app_task_cancelled_first_in_stop_and_output_is_bridged() -> None:
    from xr_ai_voice import VOICE_OUTPUT_TOPIC

    app = make_app()
    voice = FakeVoice()
    app._voice_factory = lambda _app: voice
    bridged: list[ResponseOutput] = []

    class VoiceSink(RuntimeAgent):
        @subscribe(VOICE_OUTPUT_TOPIC)
        async def sink(self, message: ResponseOutput, ctx: RuntimeContext) -> None:
            bridged.append(message)

    app._runtime.register("voice-sink", VoiceSink())
    await app.start()
    await asyncio.wait_for(voice.running.wait(), 1)
    await join(app)
    await app._participants["p1"].say("hi")
    assert [m.text for m in bridged] == ["hi"]
    await app.stop()
    assert voice.cancelled and app._app_tasks == set()


async def test_a_voice_agent_that_subscribes_to_voice_output_receives_spoken_output() -> None:
    """The voice agent is registered with the runtime, so what a participant says reaches its own subscription."""
    from xr_ai_voice import VOICE_OUTPUT_TOPIC

    heard: list[str] = []

    class SubscribingVoice(RuntimeAgent):
        async def run(self, runtime: object) -> None:
            await asyncio.sleep(10)

        @subscribe(VOICE_OUTPUT_TOPIC)
        async def output(self, message: ResponseOutput, ctx: RuntimeContext) -> None:
            heard.append(message.text)

    app = make_app()
    app._voice_factory = lambda _app: SubscribingVoice()
    await app.start()
    await join(app)
    await app._participants["p1"].say("hello there")
    await app.stop()
    assert heard == ["hello there"]


def test_start_stop_and_test_kit_install_no_signal_handlers_and_run_is_the_only_one_that_does() -> None:
    seen: dict[str, object] = {}
    app = make_app()

    class Probe(Component):
        async def on_start(self) -> None:
            seen["during_start"] = signal.getsignal(signal.SIGTERM)

        async def on_participant_enrolled(self, p: Participant) -> None:
            seen["handlers_in_run"] = signal.getsignal(signal.SIGTERM)

    app.add(Probe())
    original = signal.getsignal(signal.SIGTERM)

    async def lifecycle() -> None:
        await app.start()
        await app.stop()

    asyncio.run(lifecycle())
    assert seen["during_start"] == original and signal.getsignal(signal.SIGTERM) == original

    app2 = make_app()
    log: list[str] = []
    app2.add(Recorder("r", log))

    class Killer(Component):
        async def on_start(self) -> None:
            asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)

    app2.add(Killer())
    app2.run([])
    assert log == ["r:start", "r:stop"]
    assert signal.getsignal(signal.SIGTERM) == original  # removed again after the run


def test_run_writes_the_ready_file_once_started(tmp_path: Path) -> None:
    app = make_app()
    ready = tmp_path / "sub" / "ready"

    class Killer(Component):
        async def on_stop(self) -> None:
            seen.append(ready.exists())

    seen: list[bool] = []
    app.add(Killer())

    async def stop_soon() -> None:
        await asyncio.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)

    class Starter(Component):
        async def on_start(self) -> None:
            asyncio.get_running_loop().create_task(stop_soon())  # noqa: RUF006 - test helper, ends with the loop

    app.add(Starter())
    app.run(["--ready-file", str(ready)])
    assert seen == [True]


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "conf" / "app.yaml"
    path.parent.mkdir()
    path.write_text(textwrap.dedent(body))
    return path


def test_from_config_parses_resolves_relative_paths_and_starts_nothing(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
        profile: remote
        artifacts_dir: out
        voice:
          enabled: false
          voice_gate_yaml: gate.yaml
        timeouts: {drain_s: 1.5, hook_s: 2, close_s: 3}
        """,
    )
    app = App.from_config(path, enrollment=PerConnection(admit=None))
    cfg = app._config
    assert cfg.config_dir == path.parent.resolve() and cfg.profile == "remote"
    assert cfg.artifacts_dir == path.parent.resolve() / "out"
    assert cfg.voice.enabled is False and cfg.voice.voice_gate_yaml == path.parent.resolve() / "gate.yaml"
    assert cfg.timeouts == ScopeTimeouts(drain_s=1.5, hook_s=2.0, close_s=3.0)
    assert app._runtime.running is False


def test_from_config_defaults_and_rejections(tmp_path: Path) -> None:
    app = App.from_config(write_config(tmp_path, "{}"), enrollment=PerConnection(admit=None))
    assert app._config.timeouts == ScopeTimeouts()
    (tmp_path / "conf" / "bad.yaml").write_text("artifact_dir: x\n")
    with pytest.raises(ValueError, match="unknown keys"):
        App.from_config(tmp_path / "conf" / "bad.yaml", enrollment=PerConnection(admit=None))
    (tmp_path / "conf" / "bad2.yaml").write_text("timeouts: {sleep_s: 1}\n")
    with pytest.raises(ValueError, match="timeouts"):
        App.from_config(tmp_path / "conf" / "bad2.yaml", enrollment=PerConnection(admit=None))
    (tmp_path / "conf" / "bad3.yaml").write_text("- a\n")
    with pytest.raises(ValueError, match="mapping"):
        App.from_config(tmp_path / "conf" / "bad3.yaml", enrollment=PerConnection(admit=None))


def test_import_xr_ai_does_not_load_the_voice_stack() -> None:
    import subprocess

    import xr_ai_app

    out = subprocess.run(
        [sys.executable, "-c", "import sys, xr_ai_app; print('pipecat' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert out.stdout.strip().splitlines()[-1] == "False"
    assert issubclass(xr_ai_app.App, App)


def test_run_exits_even_when_a_hook_ignores_cancellation(tmp_path: Path) -> None:
    import subprocess

    script = tmp_path / "hang.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio, os, signal
            from pathlib import Path
            from xr_ai_app import Component
            from xr_ai_app._base.app import App
            from xr_ai_app._base.app import AppConfig
            from xr_ai_app._base.scope import ScopeTimeouts

            class Stubborn(Component):
                async def on_start(self):
                    asyncio.get_running_loop().call_later(0.1, os.kill, os.getpid(), signal.SIGTERM)

                async def on_stop(self):
                    while True:
                        try:
                            await asyncio.sleep(10)
                        except asyncio.CancelledError:
                            pass

            app = App(AppConfig(config_dir=Path("."), timeouts=ScopeTimeouts(hook_s=0.1)))
            app.add(Stubborn())
            app.run([])
            print("exited")
            """
        )
    )
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=30, check=False)
    assert out.stdout.strip().endswith("exited"), out.stderr[-500:]


class StoppableVoice:
    """Voice stand-in with the app's stop seam: ``run`` ends when ``request_stop`` is called."""

    def __init__(self) -> None:
        self.stop_requests = 0
        self.cancelled = False
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self.stop_requests += 1
        self._stop.set()

    async def run(self, runtime: object) -> None:
        try:
            await self._stop.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def test_stop_asks_voice_to_stop_itself_before_cancelling_anything() -> None:
    app = make_app()
    voice = StoppableVoice()
    app._voice_factory = lambda _app: voice
    await app.start()
    await app.stop()
    assert voice.stop_requests == 1 and voice.cancelled is False
    assert app._app_tasks == set()


async def test_stop_requested_inside_startup_cancels_the_start_without_a_cycle():
    app = make_app()

    @app.on_start
    async def stop_from_hook():
        await app.stop()

    with pytest.raises(RuntimeError, match="app stopped during start"):
        await asyncio.wait_for(app.start(), 2)
    await asyncio.wait_for(app.stop(), 2)
    assert app._start_task.done()
    assert app._stop_task.done()
    assert not app._admitting
