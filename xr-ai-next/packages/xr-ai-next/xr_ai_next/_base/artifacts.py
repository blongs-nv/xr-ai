# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped artifact writer."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ArtifactSink:
    """Async JSONL and blob writer that drains when its participant leaves.

    Writes run on a worker thread, one after another, so file I/O never blocks the event loop and
    ordering is preserved. After :meth:`drain` writes are silent no-ops (rule R6.1): a task that is
    already being cancelled never raises into app code.

    A ``name`` may contain ``/`` to nest files in directories (``"views/step_2"``): each segment is
    sanitized, empty and ``.`` segments are dropped, a ``..`` segment raises ``ValueError``, and the
    directories are created on write. Every name stays inside :attr:`directory`.
    """

    def __init__(self, directory: Path, *, participant_id: str) -> None:
        self._directory = Path(directory) / _SAFE_NAME.sub("_", participant_id)
        self._participant_id = participant_id
        self._lock = asyncio.Lock()
        self._pending: set[asyncio.Task[object]] = set()
        self._closed = False

    @property
    def directory(self) -> Path:
        """Directory holding this participant's artifacts."""
        return self._directory

    @property
    def pending(self) -> bool:
        """Whether an accepted write or read is still running (read-only).

        ``App._is_idle``, and so ``TestSession.settle()``, consults it: a task parked on this sink's
        worker thread is not idle.
        """
        return bool(self._pending)

    def _path(self, name: str, suffix: str = "") -> Path:
        segments = name.split("/")
        if ".." in segments:
            raise ValueError(f"artifact name {name!r} must not contain '..'")
        safe = [clean for part in segments if (clean := _SAFE_NAME.sub("_", part).strip("."))] or ["artifact"]
        if suffix and not safe[-1].endswith(suffix):
            safe[-1] += suffix
        return self._directory.joinpath(*safe)

    async def _submit(self, work: Any) -> Any:
        # Each write is a task, so drain() can await every accepted write even if its caller is cancelled.
        task = asyncio.ensure_future(self._serialized(work))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return await asyncio.shield(task)

    async def _serialized(self, work: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(work)

    async def write_jsonl(self, name: str, record: Mapping[str, Any] | BaseModel) -> None:
        """Append one JSON line to ``<directory>/<name>.jsonl``."""
        path = self._path(name, ".jsonl")
        if self._closed:
            return
        data = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        line = json.dumps(data, ensure_ascii=False, default=str) + "\n"

        def work() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)

        await self._submit(work)

    async def write_blob(
        self, name: str, data: bytes, *, content_type: str | None = None, replace: bool = False
    ) -> Path:
        """Write ``data`` under ``name`` and return its path.

        With ``replace=True`` the bytes go to a temporary file in the same directory that is then
        renamed over the target, so a reader never sees a half-written file. ``content_type`` is
        accepted for sinks that keep metadata; the file sink stores bytes only. After the sink
        closed, nothing is written and the would-be path is returned.
        """
        path = self._path(name)
        if self._closed:
            return path

        def work() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not replace:
                path.write_bytes(data)
                return
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

        await self._submit(work)
        return path

    async def read_jsonl(self, name: str, *, last: int | None = None) -> list[dict[str, Any]]:
        """Read back ``<name>.jsonl``: every record, or only the final ``last`` ones.

        Runs after the writes already accepted, so it sees them. A missing file is an empty list.
        Works after :meth:`drain` too (reading is not a write).
        """
        path = self._path(name, ".jsonl")

        def work() -> list[dict[str, Any]]:
            if not path.is_file():
                return []
            with path.open(encoding="utf-8") as handle:
                lines = [line for line in handle if line.strip()]
            if last is not None:
                lines = lines[-last:] if last > 0 else []
            return [json.loads(line) for line in lines]

        return await self._submit(work)

    async def flush(self) -> None:
        """Wait for every accepted write to finish without closing the sink."""
        pending = tuple(self._pending)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def drain(self) -> None:
        """Flush every pending write; called by the un-enroll sequence. Later writes are no-ops."""
        self._closed = True
        pending = tuple(self._pending)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


__all__ = ["ArtifactSink"]
