# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifacts: nested names, atomic replace, readback, flush and the app-level sink."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from xr_ai_next._base.app import App, AppConfig
from xr_ai_next._base.artifacts import ArtifactSink


async def test_nested_names_create_directories_and_stay_inside_the_sink(tmp_path: Path) -> None:
    sink = ArtifactSink(tmp_path, participant_id="p1")
    await sink.write_jsonl("steps/one", {"n": 1})
    path = await sink.write_blob("views/a b/c.txt", b"x")
    assert (sink.directory / "steps" / "one.jsonl").is_file()
    assert path == sink.directory / "views" / "a_b" / "c.txt" and path.read_bytes() == b"x"
    odd = await sink.write_blob("/./.hidden//n", b"y")  # empty and dot segments are dropped
    assert odd == sink.directory / "hidden" / "n"
    for bad in ("../escape", "a/../../b", ".."):
        with pytest.raises(ValueError, match="'..'"):
            await sink.write_blob(bad, b"no")
        with pytest.raises(ValueError, match="'..'"):
            await sink.write_jsonl(bad, {})
    assert not (tmp_path / "escape").exists()


async def test_replace_swaps_the_file_atomically_and_leaves_no_temporary(tmp_path: Path) -> None:
    sink = ArtifactSink(tmp_path, participant_id="p1")
    path = await sink.write_blob("view.html", b"one", replace=True)
    await sink.write_blob("view.html", b"two", replace=True)
    assert path.read_bytes() == b"two"
    assert [f.name for f in path.parent.iterdir()] == ["view.html"]


async def test_read_jsonl_sees_accepted_writes_and_returns_the_last_n(tmp_path: Path) -> None:
    sink = ArtifactSink(tmp_path, participant_id="p1")
    assert await sink.read_jsonl("log") == []
    await asyncio.gather(*(sink.write_jsonl("log", {"i": i}) for i in range(5)))
    assert [r["i"] for r in await sink.read_jsonl("log")] == [0, 1, 2, 3, 4]
    assert [r["i"] for r in await sink.read_jsonl("log", last=2)] == [3, 4]
    assert await sink.read_jsonl("log", last=0) == []
    await sink.drain()
    assert len(await sink.read_jsonl("log")) == 5  # reading works after drain


async def test_flush_waits_without_closing(tmp_path: Path) -> None:
    sink = ArtifactSink(tmp_path, participant_id="p1")
    writes = [asyncio.ensure_future(sink.write_jsonl("log", {"i": i})) for i in range(3)]
    await asyncio.sleep(0)
    await sink.flush()
    assert len((sink.directory / "log.jsonl").read_text().splitlines()) == 3
    await sink.write_jsonl("log", {"i": 3})  # still open
    assert len(await sink.read_jsonl("log")) == 4
    await asyncio.gather(*writes)


async def test_pending_property_tracks_accepted_writes(tmp_path: Path) -> None:
    import threading

    sink = ArtifactSink(tmp_path, participant_id="p1")
    assert sink.pending is False
    gate = threading.Event()
    task = asyncio.ensure_future(sink._submit(gate.wait))
    await asyncio.sleep(0.01)
    assert sink.pending is True
    gate.set()
    await task
    await asyncio.sleep(0)
    assert sink.pending is False


async def test_settle_and_is_idle_wait_for_participant_sink_writes(tmp_path: Path) -> None:
    """A task parked on a sink thread write is not idle, so settle() waits for it."""
    import threading

    app = App(AppConfig(config_dir=tmp_path, artifacts_dir=tmp_path / "out"))
    async with app.test_session() as s:
        await s.join()
        p = app._participants["p1"]
        for sink in (p.artifacts,):
            gate = threading.Event()
            done: list[bool] = []

            async def write(sink=sink, gate=gate, done=done) -> None:
                await sink._submit(gate.wait)
                done.append(True)

            p.spawn(write(), key="turn")
            await asyncio.sleep(0.01)
            assert app._is_idle() is False
            asyncio.get_running_loop().call_later(0.05, gate.set)
            await s.settle()
            assert done == [True] and app._is_idle() is True
