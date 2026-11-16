# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frames (``FrameStream``) on FakeHub."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from xr_ai_next._base.app import App, AppConfig
from xr_ai_next._base.enrollment import PerConnection
from xr_ai_next._base.frames import Frame, FrameStream


def make_app() -> App:
    return App(AppConfig(config_dir=Path(".")), enrollment=PerConnection(admit=None))


def frame(seq: int, age_s: float = 0.0) -> Frame:
    return Frame(b"x", time.time_ns() // 1000 - int(age_s * 1e6), 8, 8, sequence=seq)


# ------------------------------------------------------------------ FrameStream


async def test_latest_next_and_age() -> None:
    fs = FrameStream()
    assert await fs._fresh() is None
    assert await fs._next(timeout_s=0.01) is None
    fs._push(frame(1, age_s=10))
    assert await fs._fresh(max_age_s=2.0) is None
    assert (await fs._fresh()).sequence == 1
    assert await fs._next(timeout_s=0.01) is None  # latest already returned it
    asyncio.get_running_loop().call_later(0.01, fs._push, frame(2))
    assert (await fs._next(timeout_s=1)).sequence == 2


async def test_sample_skips_duplicates_and_rate_limits() -> None:
    fs = FrameStream()
    seen: list[int] = []

    async def reader() -> None:
        async for f in fs.every(0.05):
            seen.append(f.sequence)

    task = asyncio.create_task(reader())
    for i in range(1, 20):
        fs._push(frame(i))
        await asyncio.sleep(0.01)
    fs._close()
    await asyncio.wait_for(task, 1)
    assert seen[0] == 1 and seen == sorted(set(seen))
    assert 2 <= len(seen) < 10  # about 4 at 50 ms over 190 ms, never every frame


async def test_close_releases_frame_and_ends_readers() -> None:
    fs = FrameStream()
    fs._push(frame(1))
    fs._close()
    assert await fs._fresh() is None
    assert await fs._next(timeout_s=5) is None
    assert [f async for f in fs.every(0)] == []


async def test_frames_follow_participant_scope() -> None:
    app = make_app()
    async with app.test_session() as s:
        await s.join()
        await s.show_frame(s.hub.make_frame())
        p = s.participant()
        fs = p.frames
        assert (await fs._fresh()).sequence == 1
        await s.leave()
        assert await fs._fresh() is None


async def test_sample_skips_stale_and_repeated_frames() -> None:
    fs = FrameStream()
    fs._push(frame(1, age_s=30))  # a stale first frame is never yielded
    seen: list[int] = []

    async def reader() -> None:
        async for f in fs.every(0):
            seen.append(f.sequence)

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.02)
    assert seen == []
    for seq in (2, 2, 3, 3):  # the same sequence pushed again is not distinct
        fs._push(frame(seq))
        await asyncio.sleep(0.02)
    fs._close()
    await asyncio.wait_for(task, 1)
    assert seen == [2, 3]


async def test_every_without_a_stream_asks_for_a_picture_each_tick() -> None:
    fs = FrameStream()
    asked: list[float] = []

    async def picture() -> Frame:
        asked.append(asyncio.get_running_loop().time())
        return frame(0)  # a picture is not a streamed frame: no sequence

    fs._capture = picture
    got: list[Frame] = []
    async for f in fs.every(0.05):
        got.append(f)
        if len(got) == 3:
            break
    assert len(asked) == 3
    assert asked[1] - asked[0] >= 0.045 and asked[2] - asked[1] >= 0.045  # the tick cadence is kept
    assert asked[1] - asked[0] < 1.0  # once on demand, the next tick asks at once (no wait for the stream)


async def test_every_with_a_stream_does_not_ask_for_pictures() -> None:
    fs = FrameStream()
    asked: list[int] = []

    async def picture() -> Frame:
        asked.append(1)
        return frame(0)

    fs._capture = picture
    seen: list[int] = []

    async def reader() -> None:
        async for f in fs.every(0.02):
            seen.append(f.sequence)

    task = asyncio.create_task(reader())
    for i in range(1, 15):
        fs._push(frame(i))
        await asyncio.sleep(0.01)
    fs._close()
    await asyncio.wait_for(task, 1)
    assert asked == [] and seen[0] == 1 and seen == sorted(set(seen))


async def test_every_goes_on_after_a_picture_that_does_not_come() -> None:
    fs = FrameStream()
    calls: list[int] = []

    async def picture() -> Frame:
        calls.append(1)
        if len(calls) == 1:
            await asyncio.sleep(10)  # times out
        return frame(0)

    fs._capture = picture
    started = asyncio.get_running_loop().time()
    gen = fs.every(0.01, timeout_s=0.05)
    f = await asyncio.wait_for(anext(gen), 5)
    await gen.aclose()
    assert f.sequence == 0 and len(calls) == 2
    # the first tick waited for the stream (2 s, the stale age), asked, timed out; after a picture that timed out the
    # next tick asks at once instead of waiting for the stream again
    assert asyncio.get_running_loop().time() - started < 2.5


def test_every_on_demand_keeps_its_cadence_on_the_virtual_clock() -> None:
    from xr_ai_next.devkit import run_virtual  # noqa: PLC0415

    async def ticks() -> list[float]:
        fs = FrameStream()

        async def picture() -> Frame:
            await asyncio.sleep(0.2)  # the device takes a moment
            return frame(0)

        fs._capture = picture
        loop = asyncio.get_running_loop()
        times: list[float] = []
        async for _ in fs.every(1.0):
            times.append(loop.time())
            if len(times) == 4:
                break
        return times

    started = time.monotonic()
    times = run_virtual(ticks())
    gaps = [round(b - a, 1) for a, b in zip(times, times[1:], strict=False)]
    assert gaps == [1.2, 1.2, 1.2]  # the interval, then the picture's own time
    assert time.monotonic() - started < 2  # virtual: idle waits skipped
