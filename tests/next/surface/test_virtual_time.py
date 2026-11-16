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
from pathlib import Path

import pytest
from xr_ai_next._base import _virtual
from xr_ai_next.devkit import exercise, resolve_target, run_virtual
from xr_ai_next.testing import FakeModels

BED = {"seen": "The bed has a spool on it.", "state": "objects_on_it"}
CLEAR = {"seen": "The bed is clear.", "state": "clear"}


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


def test_a_step_agrees_on_both_clocks_and_virtual_is_fast(load_sample, clip: Path) -> None:
    runs = {}
    for name, run in (("real", asyncio.run), ("virtual", run_virtual)):
        m = load_sample("10-guided-procedure")
        models = FakeModels({"perceive": [BED, CLEAR, CLEAR, CLEAR, CLEAR]})
        unit = resolve_target(m.app, "step:bed_clear")
        runs[name] = _timed(run, exercise(m.app, unit, video=clip, from_s=2.0, to_s=8.0, fps=2, models=models))
    (real, real_wall), (virtual, virtual_wall) = runs["real"], runs["virtual"]
    assert real.outcome == virtual.outcome == "Done", (real.to_json(), virtual.to_json())
    assert abs(real.ended_at_s - virtual.ended_at_s) <= 0.5  # within one frame interval at 2 fps
    assert [t for _, t in real.reports] == [t for _, t in virtual.reports]
    assert virtual_wall < real_wall / 3, (real_wall, virtual_wall)


def test_a_ninety_second_timer_step_runs_in_moments(load_sample) -> None:
    m = load_sample("10-guided-procedure")
    unit = resolve_target(m.app, "step:warm_up")  # Timer(90)
    run, wall = _timed(run_virtual, exercise(m.app, unit, models=FakeModels({})))
    assert run.outcome == "Done" and run.result.value == 90
    assert run.reports[-1][1] == "time is up"
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


def test_real_http_model_calls_keep_their_latency_and_timeouts(
    load_sample, clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_sample("10-guided-procedure")  # sets the guard profile: every role at a closed port
    guard = json.loads(Path(__import__("os").environ["XR_AI_PROFILE"]).read_text())
    runs = {}
    for name, run in (("real", asyncio.run), ("virtual", run_virtual)):
        url, seen, server = _stub_vision_server(0.4, [BED, CLEAR, CLEAR, CLEAR, CLEAR, CLEAR])
        try:
            profile = json.loads(json.dumps(guard))
            profile["models"]["perceive"]["endpoint"]["base_url"] = url  # only perceive is real; nothing else reachable
            path = tmp_path / f"stub_{name}.json"
            path.write_text(json.dumps(profile))
            monkeypatch.setenv("XR_AI_PROFILE", str(path))
            m = load_sample("10-guided-procedure")
            unit = resolve_target(m.app, "step:bed_clear")
            result, wall = _timed(run, exercise(m.app, unit, video=clip, from_s=2.0, to_s=8.0, fps=2, models="real"))
        finally:
            server.should_exit = True
        failed = [e for e in result.events if e["kind"] in ("error", "procedure") and "fail" in json.dumps(e)]
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
