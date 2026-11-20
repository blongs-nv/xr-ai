# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The virtual-time loop: idle waits are skipped; threads, held work and real model calls still take real time.

Each comparison runs the same thing on the real clock (``asyncio.run``) and on the virtual one (``run_virtual``):
the outcome and the clip time it ended at must agree, and the virtual run must be much faster. The last test makes
real HTTP model calls (a local OpenAI-compatible stub in its own thread, with latency) to show their timeouts are not
hit early and their latency is kept.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel
from xr_ai_app._base import _virtual
from xr_ai_app.devkit import LatencyProfile, exercise, run_virtual
from xr_ai_app.testing import FakeModels
from xr_ai_kit.sop import Run

BED = "A spool lies on the print bed."
CLEAR = "The print bed is in view and empty."
BED_READ = {"seen": "The bed has a spool on it.", "state": "objects_on_it"}
CLEAR_READ = {"seen": "The bed is clear.", "state": "clear"}


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cv2 = pytest.importorskip("cv2")
    import numpy as np  # noqa: PLC0415

    path = tmp_path_factory.mktemp("clip") / "demo.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (64, 48))
    for _ in range(40):
        writer.write(np.full((48, 64, 3), 128, dtype=np.uint8))
    writer.release()
    return path


def step_unit(step):  # noqa: ANN001, ANN201
    """A unit that runs one step of sample 10 on its own, with a run as its io (as the runner would)."""

    async def unit(p, probe):  # noqa: ANN001, ANN202
        return await step.run(Run(p, 1, name="exercise"))

    return unit


def _timed(run, coro):  # noqa: ANN001, ANN202
    started = time.monotonic()
    result = run(coro)
    return result, time.monotonic() - started


def test_idle_waits_are_skipped_real_work_is_not() -> None:
    async def scenario() -> dict[str, float]:
        loop = asyncio.get_running_loop()
        marks = {}
        t, w = loop.time(), time.monotonic()
        await asyncio.sleep(90)
        marks["idle"] = (loop.time() - t, time.monotonic() - w)
        t, w = loop.time(), time.monotonic()
        await asyncio.to_thread(time.sleep, 0.3)
        marks["thread"] = (loop.time() - t, time.monotonic() - w)
        t, w = loop.time(), time.monotonic()

        async def model_call() -> None:
            with _virtual.held():  # what every real role call does
                await asyncio.sleep(0.3)

        await asyncio.wait_for(model_call(), 5)  # its timeout must not be reached by skipping
        marks["held"] = (loop.time() - t, time.monotonic() - w)
        marks["wall_offset"] = (_virtual.wall_us() - time.time_ns() // 1000) / 1e6
        return marks

    marks = run_virtual(scenario())
    assert marks["idle"][0] == pytest.approx(90, abs=0.01) and marks["idle"][1] < 0.2
    assert marks["thread"][0] == pytest.approx(0.3, abs=0.1) and marks["thread"][1] >= 0.29
    assert marks["held"][0] == pytest.approx(0.3, abs=0.1) and marks["held"][1] >= 0.29
    assert marks["wall_offset"] == pytest.approx(90, abs=0.2)  # frame ages move with the loop
    assert asyncio.run(asyncio.sleep(0, result=_virtual.wall_us())) - time.time_ns() // 1000 < 1_000_000  # normal loop


class _Role:
    """A role whose call is real work in a thread (``real_s`` here), held like a real role's model call."""

    def __init__(self, name: str) -> None:
        self.name = name

    @_virtual.held_role_call
    async def ask(self, real_s: float) -> None:
        await asyncio.to_thread(time.sleep, real_s)


def test_a_latency_profile_charges_each_role_call_its_declared_time_however_long_it_took_here() -> None:
    async def scenario() -> tuple[float, float, float, dict[str, list[float]]]:
        loop = asyncio.get_running_loop()
        perceive = _Role("perceive")
        t, wall = loop.time(), time.monotonic()
        await perceive.ask(0.05)  # faster here than declared
        faster, faster_wall = loop.time() - t, time.monotonic() - wall
        t = loop.time()
        await perceive.ask(0.3)  # slower here than declared
        slower = loop.time() - t
        return faster, faster_wall, slower, loop.real_latencies_s

    faster, faster_wall, slower, measured = run_virtual(scenario(), latency=LatencyProfile({"perceive": 0.2}))
    assert faster == pytest.approx(0.2, abs=1e-6) and faster_wall < 0.2
    assert slower == pytest.approx(0.2, abs=1e-6)

    async def wall_drift_s() -> float:
        return (_virtual.wall_us() - time.time_ns() // 1000) / 1e6

    assert abs(run_virtual(wall_drift_s(), latency=LatencyProfile.zero())) < 1.0  # wall time starts at the real one
    assert len(measured["perceive"]) == 2 and measured["perceive"][1] >= 0.29  # what it really took, recorded


def test_under_a_latency_profile_timers_keep_firing_while_a_call_runs() -> None:
    async def scenario() -> tuple[list[float], float]:
        loop = asyncio.get_running_loop()
        started, ticks = loop.time(), []

        async def camera() -> None:  # a look every second, as a watch step's loop
            while True:
                await asyncio.sleep(1)
                ticks.append(round(loop.time() - started, 6))

        looking = asyncio.create_task(camera())
        await _Role("answer_with_reasoning").ask(0.1)  # declared 5 s: the escalator reasoning
        took = loop.time() - started
        looking.cancel()
        return ticks, took

    ticks, took = run_virtual(scenario(), latency=LatencyProfile({"answer_with_reasoning": 5.0}))
    assert took == pytest.approx(5.0, abs=1e-6)
    assert ticks[:4] == [1.0, 2.0, 3.0, 4.0]  # the camera kept looking while the model reasoned


@pytest.mark.parametrize("end_s", [56.09, 345.1, 12.7])
def test_under_a_latency_profile_waiting_out_the_time_left_ends(end_s: float) -> None:
    """What an evaluation does at a scene's end: wait for the time left. The clock must reach the end (it once
    stopped a hair short, and the wait repeated forever)."""

    async def scenario() -> float:
        loop = asyncio.get_running_loop()
        begun = loop.time()
        forever = asyncio.ensure_future(asyncio.Event().wait())

        async def camera() -> None:
            while True:
                await asyncio.sleep(1.87)

        looking = asyncio.ensure_future(camera())
        while (remaining := end_s - (loop.time() - begun)) > 0:
            await asyncio.wait({forever}, timeout=remaining)
        forever.cancel()
        looking.cancel()
        return loop.time() - begun

    ended: list[float] = []
    runner = threading.Thread(
        target=lambda: ended.append(run_virtual(scenario(), latency=LatencyProfile.zero())), daemon=True
    )
    runner.start()
    runner.join(5)
    assert not runner.is_alive(), "the wait for the time left never ended"
    assert ended[0] == pytest.approx(end_s, abs=1e-6)


def test_a_latency_profile_loads_from_yaml_scales_and_names_every_role(tmp_path: Path) -> None:
    (tmp_path / "spark.yaml").write_text("perceive: 1.5\ndecide: 0.2\n", encoding="utf-8")
    profile = LatencyProfile.load(tmp_path / "spark.yaml", scale=2.0)
    assert (profile.seconds_for("perceive"), profile.seconds_for("decide"), profile.seconds_for(None)) == (3.0, 0.4, 0)
    with pytest.raises(KeyError, match="answer_with_reasoning"):
        profile.seconds_for("answer_with_reasoning")  # a role the profile misses is an error, never silent
    assert LatencyProfile.zero().seconds_for("anything") == 0.0
    (tmp_path / "bad.yaml").write_text("perceive: fast\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maps role names"):
        LatencyProfile.load(tmp_path / "bad.yaml")


def test_a_step_agrees_on_both_clocks_and_virtual_is_fast(load_sample, clip: Path) -> None:
    runs = {}
    for name, run in (("real", asyncio.run), ("virtual", run_virtual)):
        m = load_sample("10-guided-procedure")
        models = FakeModels({"perceive": [BED, CLEAR, CLEAR], "decide": ["negative", "positive", "positive"]})
        unit = step_unit(m.bed_clear)
        runs[name] = _timed(run, exercise(m.app, unit, video=clip, from_s=2.0, to_s=8.0, fps=2, models=models))
    (real, real_wall), (virtual, virtual_wall) = runs["real"], runs["virtual"]
    assert real.outcome == virtual.outcome == "Verification", (real.to_json(), virtual.to_json())
    assert abs(real.ended_at_s - virtual.ended_at_s) <= 0.5  # within one frame interval at 2 fps
    labels = [[j.label for j in r.result.judgments] for r in (real, virtual)]
    assert labels[0] == labels[1] == ["negative", "positive", "positive"]
    assert virtual_wall < real_wall / 3, (real_wall, virtual_wall)


def test_a_ninety_second_timer_step_runs_in_moments(load_sample) -> None:
    m = load_sample("10-guided-procedure")
    unit = step_unit(m.warm_up)  # a 90 s wait
    run, wall = _timed(run_virtual, exercise(m.app, unit, models=FakeModels({})))
    assert run.outcome == "TimerState" and run.result.status == "completed"
    assert run.result.duration_s == 90
    assert wall < 5, wall


# ------------------------------------------------------------------------------------ real model calls


def _stub_vision_server(latency_s: float, replies: list[dict]) -> tuple[str, list[float], object]:  # noqa: ANN001
    """A real OpenAI-compatible HTTP server on 127.0.0.1 in its own thread (its own, normal event loop)."""
    import uvicorn  # noqa: PLC0415
    from starlette.applications import Starlette  # noqa: PLC0415
    from starlette.responses import JSONResponse  # noqa: PLC0415
    from starlette.routing import Route  # noqa: PLC0415

    seen: list[float] = []

    async def chat(request) -> JSONResponse:  # noqa: ANN001
        await request.body()
        seen.append(time.monotonic())
        await asyncio.sleep(latency_s)  # real latency: this loop is not virtual
        reply = replies[min(len(seen) - 1, len(replies) - 1)]
        message = {"role": "assistant", "content": json.dumps(reply)}
        return JSONResponse({"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]})

    async def models(request) -> JSONResponse:  # noqa: ANN001
        return JSONResponse({"data": [{"id": "stub"}]})

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/v1/models", models),
            Route("/health", models),
        ]
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    assert server.started
    return f"http://127.0.0.1:{port}", seen, server


class Bed(BaseModel):
    seen: str
    state: Literal["clear", "objects_on_it", "cannot_tell"]


@dataclass
class Done:
    looks: int


def bed_watch(perceive):  # noqa: ANN001, ANN201
    """A camera check on one real role: a structured read a second until two in a row say clear."""

    async def unit(p, probe):  # noqa: ANN001, ANN202
        looks, streak = 0, 0
        async for frame in p.frames.every(1.0):
            looks += 1
            bed = await perceive.ask(frame, "Is the print bed clear?", response_format=Bed, timeout_s=5.0)
            streak = streak + 1 if bed.state == "clear" else 0
            if streak == 2:
                return Done(looks)
        return None

    return unit


def test_real_http_model_calls_keep_their_latency_and_timeouts(
    load_sample, clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_sample("10-guided-procedure")  # sets the guard profile: every role at a closed port
    guard = json.loads(Path(__import__("os").environ["XR_AI_PROFILE"]).read_text())
    runs = {}
    for name, run in (("real", asyncio.run), ("virtual", run_virtual)):
        url, seen, server = _stub_vision_server(0.4, [BED_READ, *[CLEAR_READ] * 5])
        try:
            profile = json.loads(json.dumps(guard))
            profile["models"]["perceive"]["endpoint"]["base_url"] = url  # only perceive is real; nothing else reachable
            path = tmp_path / f"stub_{name}.json"
            path.write_text(json.dumps(profile))
            monkeypatch.setenv("XR_AI_PROFILE", str(path))
            m = load_sample("10-guided-procedure")
            unit = bed_watch(m.perceive)
            result, wall = _timed(run, exercise(m.app, unit, video=clip, from_s=2.0, to_s=8.0, fps=2, models="real"))
        finally:
            server.should_exit = True
        failed = [e for e in result.events if e["kind"] == "error" and "fail" in json.dumps(e)]
        runs[name] = (result, wall, len(seen), failed)
    (real, real_wall, real_calls, real_failed), (virtual, virtual_wall, virtual_calls, virtual_failed) = (
        runs["real"],
        runs["virtual"],
    )
    assert real.outcome == virtual.outcome == "Done", (real.to_json(), virtual.to_json())
    assert not real_failed and not virtual_failed  # no model-call timeouts on either clock
    assert real_calls >= 3 and virtual_calls >= 3
    assert all(call["duration_s"] >= 0.39 for call in virtual.calls)  # latency kept: the loop waited for real
    assert abs(real.ended_at_s - virtual.ended_at_s) <= 1.0  # within one look interval
    assert virtual_wall < real_wall, (real_wall, virtual_wall)
